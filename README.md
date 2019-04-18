# LyricGenerator

## Overview
As part of the Hip Hop generation group, this project utilized machine learning techniques to generate lyrics and pair them with melodies based on sentiment analysis. An lstm was trained on lyrics to produce music lyrics, and naive bayes nets were used to classify the bars based on emotions.

## Generating Lyrics
A long short-term memory RNN created with the keras library for python machine learning was used to generate lyrics. The model was trained on a wide variety of genres of songs with songs retrieved from MetroLyrics. The line carry characters are maintained in the dataset and reproduced by the model allowing the lstm to learn logical places for bars to begin and end, and deciding where bars are split for later analysis.

## Lyrics Analysis
In order to find a good melodic fit for a particular bar of lyrics, each bar is evaluated for the number of phonemes it contains and the emotional sentiment most strongly coorelating with the content of the lyrics. 

Phonemes are counted as an estimate for the number of notes a particular bar should hold, as phonemes depict the number of particular sounds produced within every word. Phonemes are counted using the nltk package for python. 

Sentiment analysis is achieved by scoring the valence and arousal for every lyrical bar. Valence, arousal, and dominance are the three major sentiment descriptors in phsycology, and using just valence and arousal a wide variety of emotions can be described. Valence and Arousal are both classified using a naive bayes nets. The nltk VADER sentiment package is used to score valence. A naive bayes net trained on the EmoBank dataset is used to score arousal and was constructed using the nltk pakage for bayes nets.

## Technical Descrition
Here is an outline of important files in this repository:

lstm.py  -The lstm used to generate Lyrics for the project.

Lyrics.txt  -The dataset of song lyrics used to train the lstm.

LyricProcessing.py  -The tools for annotating lyrics with phonemes, and sentiment. Contains the naive bayes nets used to classify sentiment.
