import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        
# torch.manual_seed(1)

def eval(dataloader,model,loader,device, printText=False):

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

    songName = "Harmonized Excerpt"
    evaluation_helpers.viewPhrase(decoded_melody,decoded_predicted_chords,songName)

    # evaluation_helpers.viewPhrase(decoded_melody,decoded_actual_chords,songName)


def harmonize_melody(model,melody,device,loader,temp=1):
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
    output = output/temp

    probabilities = F.softmax(output[:,-1], dim=-1)

    sampled_chord = torch.multinomial(probabilities, 1)

    next_item = sampled_chord
    # print(output,output.shape)
    next_item = torch.tensor([[next_item.item()]], device=device)

    # Concatenate previous input with next chord
    sequence = torch.cat((sequence, next_item), dim=1)

 
  return [in2chord[chord] for chord in sequence.squeeze().tolist()]


def main():

    script_name = sys.argv[0]
    train_flag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == '--train' else None  
    if train_flag:
        eval_flag = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] == '--eval' else None 
    else:
        eval_flag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == '--eval' else None 

    json_config_path = "config.json"

    # get hyperparams from config file for training 
    with open(json_config_path, "r") as json_file:
        loaded_hyperparameters = json.load(json_file)


    # read songs, create dataloaders and vocab
    loader = Song_Dataloader()
    train_dataloader, test_dataloader,chord2in,in2chord,note2in, in2note = loader.load()
 
    # Using just CPU for current state of model:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    #     print ("GPU device not found, CPU used")

    device = torch.device("cpu")

    if train_flag:
        
        print("Training model...")
                         
        hidden_size = loaded_hyperparameters["hidden_size"]
        num_layers = loaded_hyperparameters["num_layers"]
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

        optimizer = torch.optim.Adam(model.parameters(),amsgrad=True,lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),amsgrad=True,lr=lr)

        trainer = Trainer(model,loss_fn=loss_fn,optimizer=optimizer,train_dataloader=train_dataloader,test_dataloader=test_dataloader,device=device)
        trainer.train(num_epochs)

        # save newly trained model
        trained_model = {'model':[model.kwargs,model.state_dict(),model.model_type]}
        
        torch.save(trained_model,'Saved_Models/trained_model.pth')
        print("Saved model")
    else:
        print("Loading pretrained model...")

        # change path to use different model (defaults to pretrained)

        main_model = torch.load('Saved_Models/pretrained_model.pth',map_location =device)
        
        model_kwargs,model_state,model_type = main_model['model']

        model = Transformer(**model_kwargs)  
        model.load_state_dict(model_state)
   
        print("Model loaded")

    if eval_flag:
        print("Evaluating model..")
        eval(train_dataloader,model,loader,device)
        print("Successfully outputed example from test set")
    
    # base mode is load pretrained model, conduct inference 
    if not eval_flag and not train_flag:

        print("Running model on input melody...")


        # harmonize melody in form of [midi note, duration in 16th notes]
        melody = [[60,4], [62,4],[64,4],[62,4],[64,4],[65,2],[67,2],[69,4],[67,4],[62,4],
        [64,4],[65,4],[65,4],[67,4],[69,2],[71,2],[72,4],[72,4],[60,4],
        [64,4],[65,4],[65,4],[67,4],[69,2],[71,2],[72,4],[72,4]]

        twinkle_melody =  [[60,4],[60,4], [67,4],[67,4],[69,4],[69,4], [67,8],[65,4],[65,4],[64,4],[64,4],[62,4],[62,4],[60,8]]
        moon_melody = [[67,16],[74,4],[72,12],[71,10],[69,2],[67,2],[65,2],[67,12],[60,4]]
        sequence = harmonize_melody(model,moon_melody,device,loader,temp=1.2)
        print(sequence)
        evaluation_helpers.viewPhrase(moon_melody,evaluation_helpers.decode_stream(sequence[1:]),playScore=True)
    

main()

