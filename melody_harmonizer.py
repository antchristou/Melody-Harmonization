import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import pandas as pd
import re
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import music21
import math
import sys

import evaluation_helpers
import Model.Transformer
import Trainer.trainer
from Model.Transformer import Transformer
from Trainer.trainer import Trainer
from song_dataloader import Song_Dataloader

# Uncomment to ensure same results for reproducibility each time the program is run
# torch.manual_seed(42)
    
def eval(dataloader,model,loader,device, printText=False):
    """
    Tests model on random 8 bar phrase from validation set. Opens output in 
    notation software for listening.

    Parameters:
    - dataloader: contains full validation set for testing 
    - model: trained harmony model
    - loader: dataloader that prepares input melody for form model expects
    - device: CPU/GPU, etc 
    - printText: if True, model prints expected and actually predicted chords to terminal window 

    Returns:
    None
    """

    in2chord, hord2in, note2in, in2note  = loader.get_vocab()
    
    # choose random datapoint for output 
    dataset = dataloader.dataset
    n_samples = len(dataset)

    # Get a random sample
    rand_inputs,rand_targets = random.choice(dataset)

    rand_inputs = torch.tensor(rand_inputs).unsqueeze(0)
    rand_targets = torch.tensor(rand_targets).unsqueeze(0)

    inputs = rand_inputs.to(device)
    targets = rand_targets.to(device)

    target_input = targets[:,:-1]
    target_expected = targets[:,1:]

    tgt_mask = model.get_tgt_mask(target_input.size(1)).to(device)


    output = model(inputs,target_input, tgt_mask)

    predicted_chords = torch.argmax(output,dim=2)
    output = output.permute(0,2,1)


    actual_chords = [in2chord[chord] for chord in rand_targets.squeeze().tolist()]
    predicted_chords =  [in2chord[chord] for chord in predicted_chords.squeeze().tolist()]
    if printText:
        print("Decoded predicted chords: ", predicted_chords)
        print("Actual chords: ", actual_chords)

    decoded_melody = [in2note[note] for note in rand_inputs.squeeze().tolist()]
    decoded_melody = evaluation_helpers.decode_stream(decoded_melody[:-1])
    decoded_actual_chords = evaluation_helpers.decode_stream(actual_chords[1:-1])
    decoded_predicted_chords = evaluation_helpers.decode_stream(predicted_chords[1:-1])
    
    # remove any extra EOS tokens if any were generated
    decoded_predicted_chords = evaluation_helpers.fixFormatting(decoded_predicted_chords)

    songName = "Harmonized Excerpt"
    evaluation_helpers.viewPhrase(decoded_melody,decoded_predicted_chords,songName)

    # Uncomment to view actual chords from excerpt 
    # evaluation_helpers.viewPhrase(decoded_melody,decoded_actual_chords,songName)


def harmonize_melody(model,melody,device,loader,temp=1,k=20):
  """
    Runs input melody through model and outputs input melody with generated harmonies. Opens notation
    software for viewing hearing output

    Parameters:
    - model: trained harmony model
    - melody: input melody (in form of list of tuples)
    - device: CPU/GPU, etc 
    - loader: dataloader that prepares input melody for form model expects
    - temp: (int) temperature value - lower = more conservtive,but more accurate, higher = more creative 
        but more chaotic and dissonant
    - k: (int) used in top k sampling to cut long tail of low probability chords

    Returns:
    list of output chords
  """

  encoded = loader.encode_melody(melody)
  in2chord, chord2in, note2in, in2note = loader.get_vocab()
  SOS_TOKEN, EOS_TOKEN = loader.get_special_chars()

  inputs = torch.tensor(encoded).to(device)
  inputs = inputs.unsqueeze(0)

  MAX_LENGTH = math.ceil((inputs.size(1)-1)/8) 
  sequence = torch.tensor([chord2in[SOS_TOKEN]],device=device)
  sequence = sequence.unsqueeze(0)

  while sequence.size(dim=1) <= MAX_LENGTH:

    tgt_mask = model.get_tgt_mask(sequence.size(1)).to(device)

    output = model(inputs,sequence,tgt_mask)

    # temperature scaling
    output = output/temp

    # top k sampling
    probabilities = evaluation_helpers.top_k_sampling(output[:,-1],k,device)

    probabilities = F.softmax(probabilities, dim=-1)

    sampled_chord = torch.multinomial(probabilities, 1)

    next_item = sampled_chord

    next_item = torch.tensor([[next_item.item()]], device=device)

    # Concatenate previous input with next chord
    sequence = torch.cat((sequence, next_item), dim=1)

 
  return [in2chord[chord] for chord in sequence.squeeze().tolist()]


def main():

    """
    Processes input. Command line arguments take following form:

    --train: trains model from scratch using hyperparameters located in config.json and saves to   
    trained_model.pth
    --eval: runs model on random excerpt from validation set. If --train present, uses just trained model.
    If not, loads pretrained model. 

    if neither --train nor --eval is set, model expects command line argument of input melody in form of list
    of tuples of form [midi note, duration in 16th notes]. If none is provided, model runs
    inference on default twinkle, twinkle little star melody
    
    """
    script_name = sys.argv[0]
    train_flag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == '--train' else None  
    # for deploying model in daw, run with --daw flag set and input melody provided
    daw_flag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == '--daw' else None  
    if train_flag:
        eval_flag = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] == '--eval' else None 
    else:
        eval_flag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == '--eval' else None 

    if not train_flag and not eval_flag:   
        # expect temperature, then k value for top_k sampling 
        temperature = sys.argv[-3] if len(sys.argv) > 2 else None 
        top_k = sys.argv[-2] if len(sys.argv) > 3 else None 

    if not eval_flag:
        # input melody string for inference should be last iff eval flag isn't set
        input_melody_string = sys.argv[-1]
    
    # in daw mode supress all output except final harmony
    if daw_flag:
        print_text = False
    else:
        print_text = True

    json_config_path = "config.json"

    # get hyperparams from config file for training 
    with open(json_config_path, "r") as json_file:
        loaded_hyperparameters = json.load(json_file)


    # read songs, create dataloaders and vocab
    loader = Song_Dataloader()
    train_dataloader, test_dataloader,chord2in,in2chord,note2in, in2note = loader.load()
 
    # Using just CPU for current state of model:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if print_text:
            print ("GPU device not found, CPU used")

    device = torch.device("cpu")

    if train_flag:
        
        print("Training model...")
                         
        hidden_size = loaded_hyperparameters["hidden_size"]
        num_layers = loaded_hyperparameters["num_layers"]
        # Note: lr value interacts with LR warmup and Adam and so 
        # will likely in practice be less than this value
        lr = loaded_hyperparameters["lr"] 
        dim_feedforward = loaded_hyperparameters["dim_feedforward"]
        num_heads = loaded_hyperparameters["num_heads"]
        dropout_p = loaded_hyperparameters["dropout_p"]
        input_embedding_dim = loaded_hyperparameters["input_embedding_dim"]
        output_embedding_dim = loaded_hyperparameters["output_embedding_dim"]
        num_epochs =  loaded_hyperparameters["num_epochs"]

        # instatiate base model for training according to json hyperparameters 
        model = Transformer(
        inputVocab=len(note2in),outputVocab=len(chord2in), input_embedding_dim=input_embedding_dim,output_embedding_dim=output_embedding_dim
        ,num_heads=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout_p=dropout_p,
        dim_feedforward=dim_feedforward)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # learning rate warmup epochs
        warmup_epochs = 7
    
        optimizer = torch.optim.Adam(model.parameters(),amsgrad=True,lr=lr)
  
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),amsgrad=True,lr=lr)

        trainer = Trainer(model,loss_fn=loss_fn,optimizer=optimizer,train_dataloader=train_dataloader,test_dataloader=test_dataloader,device=device,scheduler=None)
        # scheduler linearly increses LR for first warmup_epochs epochs
        scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0)
        trainer.scheduler = scheduler

        trainer.train(num_epochs)

        # save newly trained model
        trained_model = {'model':[model.kwargs,model.state_dict(),model.model_type]}
        
        torch.save(trained_model,'Saved_Models/trained_model.pth')
        print("Saved model")
    else:
        if print_text:
            print("Loading pretrained model...")

        # change path to use different model (defaults to pretrained)

        main_model = torch.load('Saved_Models/pretrained_model.pth',map_location =device)
        
        model_kwargs,model_state,model_type = main_model['model']

        model = Transformer(**model_kwargs)  
        model.load_state_dict(model_state)
        if print_text:
            print("Model loaded")

    if eval_flag:
        print("Evaluating model..")
        eval(train_dataloader,model,loader,device,printText=False)
        print("Successfully outputed example from test set")
    
    # base mode is load pretrained model, conduct inference 
    if not eval_flag and not train_flag:

        # Note: model will occasionally fail here rarely if it adds a superflous EOS token. 
        # Simply re-running once or twice should fix.

        # harmonize melody in form of [midi note, duration in 16th notes]
        melody = [[60,4], [62,4],[64,4],[62,4],[64,4],[65,2],[67,2],[69,4],[67,4],[62,4],
        [64,4],[65,4],[65,4],[67,4],[69,2],[71,2],[72,4],[72,4],[60,4],
        [64,4],[65,4],[65,4],[67,4],[69,2],[71,2],[72,4],[72,4]]

        twinkle_melody =  [[60,4],[60,4], [67,4],[67,4],[69,4],[69,4], [67,8],[65,4],[65,4],[64,4],[64,4],[62,4],[62,4],[60,8]]
        moon_melody = [[67,16],[74,4],[72,12],[71,10],[69,2],[67,2],[65,2],[67,12],[60,4]]
        input_melody = None

        if print_text:
            print("Running model on input melody...")
        
        try:
            # input melody is passed as string for decoding
            tuples_array = json.loads(input_melody_string)
            if isinstance(tuples_array, list) and all(isinstance(t, list) and len(t) == 2 for t in tuples_array):
                input_melody = tuples_array
            else:
                if print_text:
                    print("Error Invalid Format: enter input melody as list of tuples in form [midi note,duration]")
        except json.JSONDecodeError:
            if print_text:
                print("Input melody invalid or not present. Using default melody.")

        # if input melody is non-existant or invalid, default to twinkle twinkle little star
        if input_melody == None:
            input_melody = twinkle_melody
        else:
            melody_length = sum(t[1] for t in input_melody)
            # input melody can't be longer than 8 bars
            if melody_length > 8*16:
                if print_text:
                    print("Error: Input Melody must be 8 bars or less. Using default melody instead")
                input_melody = twinkle_melody

        # change k and temp values for inference 
        if top_k:
            k = int(top_k)
        else:
            k = 20
        no_temp = True
        if temperature:
            try:
                temperature = float(temperature)
                no_temp = False
            except ValueError:
                no_temp = True
        if no_temp:
            # default value on error
            temperature = 2.0

        sequence = harmonize_melody(model,input_melody,device,loader,temp=temperature,k=k)
        if print_text:
            print("Output Chord Sequence: ")
            print(sequence)

        if not daw_flag:
            evaluation_helpers.viewPhrase(input_melody,evaluation_helpers.decode_stream(sequence[1:]))
        if daw_flag: 
            evaluation_helpers.outputDAWPhrase(evaluation_helpers.decode_stream(sequence[1:]))
       

main()

