
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import sys
import io
import random
import argparse



"""
    modes:
    'init'      :: train without loading in weights, and write to weights
    'train'     :: load weights from weights, train, and then wright back to weights
    'sample'    :: don't train and instead print a sample
"""

parser = argparse.ArgumentParser(description="An LSTM")
parser.add_argument('mode', choices=['init', 'train', 'sample'], default='sample',
                    help= "Indicate whether program should be in 'int', 'train', or 'sample' mode")
                    #dest is mode ??
parser.add_argument('--sample', type=int, default=1000,
                     help="The number of characters to be generated when sampling. Default = 1000")
                    #maybe also allow temperature of samples to be chosen
                    #only if sample mode?
parser.add_argument('--epoch', type=int, default=3,
                    help='Indicate the number of epochs that the model should be trained on. Default = 3')
                    #only if not sample mode ??
parser.add_argument('--step', type=int, default=5,
                    help='The step size for creating sentences from corpus. Default = 5')
parser.add_argument('--sampleSize', type=int, default=(-1),
                    help="The maximum number of characters from the corpus to be trained on. Default is one million")
#parser.add_argument('--corpus', type=argparse.FileType('r'), default='Lyrics.txt',
#                   help="Indicate the corpus source file if not 'Lyrics.txt'")
parser.add_argument('--corpus', type=str, default='Lyrics.txt',
                    help="Indicate the corpus source file if not 'Lyrics.txt'")
args = parser.parse_args()

mode = args.mode
printSize = args.sample
numEpoch = args.epoch
step = args.step
sampleSize = args.sampleSize
corpus = args.corpus
weights = 'weights.hdf5'

#########################
##### Prepare Input #####
#########################

with io.open(corpus, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

data_size, vocab_size = len(text), len(chars)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
sentences = []
next_chars = []

textStart = 0
textEnd = 0
if len(text) > sampleSize and sampleSize > 0:
    textEnd = random.randint(sampleSize - 1, len(text) - 1)
    textStart = textEnd - sampleSize
    for i in range(textStart, textEnd - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
else:
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1




###########################
##### Construct Model #####
###########################

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# load weights
if mode != 'init':
    model.load_weights("weights.hdf5")

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)




#################################
##### Training and Sampling #####
#################################

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    file = open('out.txt', 'a')
    file.write('\n \n \n')

    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        file.write('----- Generating with seed: "' + sentence + '" \n')
        sys.stdout.write(generated)

        for i in range(printSize):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        file.write(generated + '\n')


checkpoint = [ModelCheckpoint(filepath=weights)]
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

callback = checkpoint
if mode == 'sample':
    callback = print_callback
    numEpoch = 1

if mode != 'sample':
    #train
    model.fit(x, y,
              batch_size=128,
              epochs=numEpoch,
              callbacks=callback
              )
else:
    #sample
    on_epoch_end(0, 5)
