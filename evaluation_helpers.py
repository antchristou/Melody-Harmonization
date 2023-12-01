import torch
import torch.nn.functional as F

import music21
from music21 import *

import copy

import re


def viewPhrase(input,output,songName="Harmonized Excerpt", playScore=False):
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
    # filter out word alter to fit with music21 expected input format TODO improve this
    chordDur[0] = re.sub(r'\balter\b', '', chordDur[0])
    chordDur[0] = re.sub(r'\badd\b', '', chordDur[0])

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
