# Melody Harmonization with Transformers 

## Overview and Motivation

lmost all genres of music from across the world have some concept of harmony, which is the process of combining different sounds into one greater whole.
Although the particular way distinct voices are blended together to form chords, chord progressions, and countermelodies may differ across time and across culture, the concept of harmony has remained critically important to music-making for hundreds if not thousands of years.

As a result of harmony's incredible importance, this model was created to help musicians as a tool for use during the compositional process.

## Prerequisites 

python >= 3.6 

numpy >= 1.19.0

pandas==1.5.3

music21 

pytorch 

a musicXML file reader of some kind (Musescore, Sibelius, lilypond etc)


## How to Run

python3 melody_harmonizer.py [--train] [--eval] 

With the train flag, the model trains from scratch a model based upon the parameters in config.json and saves it as
trained_model.pth. If this flag is not specified, a pretrained model is loaded. 

With the eval flag, the model outputs a random example from the validation set. 

If neither the eval nor train flag are set,the model expects a command line argument input melody
to attempt to harmonize. This melody must be less than 8 bars long and in the form of 
[[midi note 1, duration in 16th notes 1], [midi note 2, duration in 16th notes 2]...]'

If the provided melody is invalid or not present, a default melody is loaded and used.
An example of a valid input is: 
python3 ./melody_harmonizer.py '[[67,16],[74,4],[72,12],[71,10],[69,2],[67,2],[65,2],[67,12],[60,4]]'

## What's here 


### 1. Datasets

   Each dataset present is encoded in 16th notes frames of the form 
   [melody,chords], as all the datasets are. The melody notes are encoded as integers representing 
   pitch classes C-B in integers, with rest, SOS, and EOS also being present.
### 3. Model
   Transformer.py -- contains transformer architecture used 
### 3. Saved_Models
   pretrained_model.pth -- pretrained model that gets used when --train is not present
   trained_model.pth -- where model trained with --train flag get stored
### 4. Trainer
   trains model according to parameters in config.json if --train flag is set
### 5. Preprocessing scripts
   contains the ipynb scripts used to clean and standardize the dataset. Can be used to add additional datasets in the future

config.json -- where hyperparamters for training model when --train flag is set can be tweaked 

evaluation_helpers.py -- helper functions for outputing harmonies and other small auxiliary tasks

melody_harmonizer.py -- main driver 

song_loader.py -- loads songs, splits into training and validation sets, creates vocab, etc.

