import torch
from torch.utils.data import DataLoader,random_split
import json

REST_TOKEN = "rest"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

class Song_Dataloader:

    """
    Steps in preprocessing pipleline:
    - transpose non tranposed songs so we have complete dataset of 4/4 C major songs
    - generate vocab
    - break into 8 measure chunks with two measure overlap
    -divide into notes vs chords (inputs vs outputs)
    -one hot as 16th note frames, batch, pad etc
    """
        
    def read_songs(self):
        
        cm_path = "Datasets/CHORD_MELODY_DATASET.json"
        jazz_path = "Datasets/JAZZ_LS_DATASET.json"
        pdsa_path = "Datasets/PDSA_DATASET.json"
        wiki_path = "Datasets/WIKIFONIA_DATASET.json"

        with open(cm_path, 'r') as json_file:
            combined_chord_melody_data = json.load(json_file)


        with open(jazz_path, 'r') as json_file:
            combined_jazz_data = json.load(json_file)

        with open(pdsa_path, 'r') as json_file:
            combined_pdsa_data = json.load(json_file)

        with open(wiki_path, 'r') as json_file:
            combined_wikifonia_data = json.load(json_file)

        # generate vocabulary
        REST_TOKEN = "rest"
        SOS_TOKEN = "<SOS>"
        EOS_TOKEN = "<EOS>"

        in2note = {}
        note2in = {}

        in2chord = {}
        chord2in = {}

        for i in range(12):
            in2note[i] = i
            note2in[i] = i

        note2in["<EOS>"] = len(note2in)
        in2note[len(in2note)] = "<EOS>"

        note2in["rest"] = len(note2in)
        in2note[len(in2note)] = "rest"


        for input,output in combined_wikifonia_data:
            for chord in output:
                if chord not in chord2in:
                    chord2in[chord] = len(chord2in)
                    in2chord[len(in2chord)] = chord

        #print("vocab size after wikifonia is: ", len(chord2in))

        for input,output in combined_jazz_data:
            for chord in output:
                if chord not in chord2in:
                    chord2in[chord] = len(chord2in)
                    in2chord[len(in2chord)] = chord

        #print("vocab size after wikifonia+jazz is: ", len(chord2in))

        for input,output in combined_pdsa_data:
            for chord in output:
                if chord not in chord2in:
                    chord2in[chord] = len(chord2in)
                    in2chord[len(in2chord)] = chord

        #print("vocab size after wikifonia+jazz+pdsa is: ", len(chord2in))

        for input,output in combined_chord_melody_data:
            for chord in output:
                if chord not in chord2in:
                    chord2in[chord] = len(chord2in)
                    in2chord[len(in2chord)] = chord

        # print ouptut vocab size if uncommented: 
        #print("vocab size after wikifonia+jazz+pdsa+chord_melody is: ", len(chord2in))
        
        # Checking vocab for debug purposes
        # print(chord2in)

        # print(len(chord2in))

        # print(note2in)
        # print(in2note)

        self.in2chord = in2chord
        self.chord2in = chord2in
        self.note2in = note2in
        self.in2note = in2note

        # Checking length of datasets for debug purposes
        # print(combined_wikifonia_data[:3])

        # for i,o in combined_wikifonia_data[:3]:
        # print(len(i))
        # print(len(o))
        # print("----")
        # for i,o in combined_jazz_data[:3]:
        # print(len(i))
        # print(len(o))
        # print("----")
        # for i,o in combined_pdsa_data[:3]:
        # print(len(i))
        # print(len(o))
        # print("----")
        # for i,o in combined_chord_melody_data[:3]:
        # print(len(i))
        # print(len(o))

        # combine datasets
        combined_data = combined_jazz_data+combined_wikifonia_data+combined_pdsa_data+combined_chord_melody_data


        # print("Total 8 measure chunks of data read:", len(combined_data))
       
        return combined_data, chord2in,in2chord,note2in, in2note

    def create_dataloaders(self,combined_data, training_split,batch_size):

        def custom_collate_fn(batch):
            input_data, output_data = zip(*batch)

            input_data = torch.tensor(input_data)
            output_data = torch.tensor(output_data)

            return input_data, output_data

        # Split into training/test set and feed into dataloader


        split_indice = int(len(combined_data)*training_split)

        training_data = combined_data[:split_indice]
        test_data = combined_data[split_indice:]

        train_dataloader = DataLoader(training_data, batch_size=batch_size,collate_fn=custom_collate_fn)
        test_dataloader =  DataLoader(test_data, batch_size=batch_size,collate_fn=custom_collate_fn)

        return train_dataloader, test_dataloader 
    
    def load(self, training_split=0.8,batch_size=128):

        # load json data and create vocab  
        combined_data, chord2in,in2chord,note2in, in2note = self.read_songs()

        # encode data 
        for input,output in combined_data:
            for i,note in enumerate(input):
                input[i] = note2in[note]
            for i,chord in enumerate(output):
                output[i] = chord2in[chord]

        # create training/test split
        train_dataloader, validation_dataloader = self.create_dataloaders(combined_data, training_split,batch_size)

        return train_dataloader,validation_dataloader,chord2in,in2chord,note2in, in2note
    
    def get_vocab(self):
        return self.in2chord, self.chord2in, self.note2in, self.in2note 


    def encode_melody(self,melody):
        """
        Transform melody into format model expects at inference time(16th note frames)
        """
        encoded = []
        for noteDur in melody:
            if noteDur[0] != "rest":
                noteName = (noteDur[0]%12)
            else:
                noteName = "rest"
            encoded_note = self.note2in[noteName]
            encoded_note = [encoded_note]*noteDur[1] # repeat the note its duration number of times (in 16th notes)
            encoded += encoded_note

        encoded.append(self.note2in[EOS_TOKEN])

        return encoded

    def get_special_chars(self):
        return SOS_TOKEN,EOS_TOKEN