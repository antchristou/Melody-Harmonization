import torch
import torch.nn.functional as F

import music21
from music21 import *

import copy

import re

import json

def outputDAWPhrase(output):
  """
  Formats phrase to be sent to DAW in form expected by API 
  """
  output = fixFormatting(output)

  # transpose output up by following value
  octaveDisplacement = 12

  notesObj = {"notes":[]}
  running_time = 0
  for chord,duration in output:
    chord = fixChordName(chord)
    chordObj =  music21.harmony.ChordSymbol(chord)

    notes = chordObj.notes

    for note in notes: 
      notesObj["notes"].append({"pitch":note.pitch.midi+octaveDisplacement,"start_time":running_time,"duration":duration*2,})
    running_time += duration*2

  notesObj = json.dumps(notesObj)
  print(notesObj)


def viewPhrase(input,output,songName="Harmonized Excerpt"):
  """
  Create viewable and audible phrase output for viewing in notation software 
  """

  phrase = stream.Part(id="melody")
  chordPhrase = stream.Part(id="chords")

  score = stream.Score(id='mainScore')

  for noteDur in input:
    if noteDur[0] == "rest": # indicates rest:
      nextNote = note.Rest()
    else:
      nextNote = note.Note(noteDur[0]+24)
    nextNote.duration.quarterLength = noteDur[1]/4.0 # divide note duration by 4 to get format useful to music21
    nextNote.octave = 5
    phrase.append(nextNote)

  currOffset = 0
  for chordDur in output:

    chordDur[0] = fixChordName(chordDur[0])

    # print(chordDur[0])
    sym = harmony.ChordSymbol(chordDur[0])
    sym.quarterLength = chordDur[1]*2
    sym = sym.transpose('P8')
    sym.writeAsChord = True
    chordPhrase.insert(currOffset,sym)

    # chord_sym = harmony.ChordSymbol(chordDur[0])
    # chord_sym.writeAsChord = False
    # phrase.insert(currOffset,chord_sym)

    currOffset += chordDur[1]*2

  score.insert(0, phrase)
  score.insert(0, chordPhrase)

  score.insert(0, metadata.Metadata())
  score.append(music21.tempo.MetronomeMark(number=120))

  score.metadata.title = songName
  score.show()


def decode_stream(melody):
  decoded_melody = []
  count = 0
  for i,note in enumerate(melody):
    if i >= 1 and note != melody[i-1]:
      decoded_melody.append([melody[i-1],count])
      count = 1
    else:
      count += 1

  decoded_melody.append([melody[-1],count])
  return decoded_melody

def top_k_sampling(logits,k,device):
  """
    Implenets top k sampling for output logits.

    Parameters:
    - logits: model output logits across entire chord vocab
    - k: (int) used in top k sampling to cut long tail of low probability chords
    - device: CPU, GPU, etc.

    Returns:
    tensor of same shape as input, but with chords outside top k zeroed out
  """

  zeros = torch.full(logits.shape, float('-inf'),device=device)

  values, indices = torch.topk(logits, k, dim=-1) # gets indices of top k most probable tokens 

  zeros.scatter_(-1, indices, values) # scatter indices of most probable tokens onto zeroed out logits
  # return top k tokens for sampling 
  return zeros


def fixChordName(chord):
  """
    Naming convention of certain complicated chords is incosistent, so this function
    tries to standardize so program doesn't crash at high temperature values.
    Since chordal vocabularly is large, some failures are possible as temperature increases
    (~ > 2) 

    This is all pretty unimportant music theory conventions/merely ensuring names match up. 
  """

   # filter out word alter to fit with music21 expected input format
  chord = re.sub(r'\balter\b', '', chord)
  chord = re.sub(r'\badd\s*b9', r'b9', chord)
  chord = re.sub(r'\badd\s*b13', r'b13', chord)
  chord = re.sub(r'\badd\s*#9', r'#9', chord)
  chord = re.sub(r'\badd\s*#11', r'#11', chord)

  # fix sus chord spelling by replacing 'sus ' with 'sus4 '
  chord = re.sub(r'sus\s', r'sus4 ', chord)
  chord = re.sub(r'sus47\s', r'sus4 ', chord)
  # Replace 'add 7 ' with '7 '
  chord = re.sub(r'add\s*7', r'7', chord)

  return chord


def fixFormatting(decoded_chords):
    """
    Due to limited data, model occasionally adds extra EOS token, which breaks model.
    This function replaces those extra values with the tonic chord. 
    Used when outputting exampels from eval set.
    """
    for chord in decoded_chords:
      if chord[0] == "<EOS>":
        chord[0] = "C"
    return decoded_chords
