import numpy
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#import csv
import pandas as pd
#nltk.download('subjectivity')
#nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer

# a list of sentences
testData = [
"You're just like poison" ,
"Slowly moving through my system" ,
"Breaking all of my defenses with time" ,
"You're just like poison and I just don't get it" ,
"How can something so deadly feel so right?" ,
"I'm not sure of what to do it's a catch 22" ,
"cause the cure is found in you I don't want it but I do" ,
"You're just like poison" ,
"My affliction, I'm addicted, I can't lie" ,
"Kiss me one more time before I die" ,
"You aint right take me high" ,
"Then that high it subsides" ,
"And my body flat lines" ,
"Then you come to revive" ,
"Wait wait wait I'm alive" ,
"But how long will it last" ,
"Will it all come crashing down?" ,
"How many doses am I needing now?" ,
"What's the prognosis will you be around?" ,
"Or am I just another victim of an assassin that broke my heart down" ,
"Baby you're just like poison" ,
"Slowly moving through my system" ,
"Breaking all of my defenses with time" ,
"You're just like poison" ,
"And I just don't get it" ,
"How can something so deadly" ,
"Feel so right" ,
"I'm not sure what to do" ,
"It's a catch 22" ,
"Cause the cure is found in you" ,
"I don't want it but I do" ,
"You're just like poison" ,
"My affliction, I'm addicted, I cant lie" ,
"Kiss me one more time before I die" ,
"It's just not my body (No)" ,
"It's my mind, you don't know" ,
"How many times I told myself" ,
"This cant do (cant do)" ,
"And that I don't need you (No I don't need you)(no)" ,
"It's so unfair(fair) that I find myself right back in your care(care)" ,
"And what's good is that when you're not always there (there there)" ,
"You're no good for my health (my heath)"
]


def paragraphToSentecnes(paragraph):
    return nltk.sent_tokenize(paragraph)

#nltk.download('cmudict')
arpabet = nltk.corpus.cmudict.dict()
def countPhonemes(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    count = 0
    for word in tokens:
        try:
            print (word, ": ", arpabet[word][0])
            count += len(arpabet[word][0])
        except Exception as e:
            print(e)
            count += 1
    print (count)
    return count

arousal_docs = []

def sentenceToFeatures(sentence):
    lem = WordNetLemmatizer()
    txt = []
    for word in nltk.word_tokenize(sentence):
        word = word.lower()
        word = lem.lemmatize(word, "v")
        if word not in stopwords.words("english"):
            txt.append(word)
            # Would be time efficent to add words to dictionary here,
            # but to keep this function more general I will not.
            #dictionary.add(word)
    return txt

def categorizeSentiment(v, a):
    sent = ""
    if v <= 0 and a <= 0:
        sent = "sad"
    if v <= 0 and a > 0:
        sent = "angry"
    if v > 0 and a <= 0:
        sent = "calm"
    if v > 0 and a > 0:
        sent = "happy"
    return sent

def addData(text, docs, val):
    try:
        txt = sentenceToFeatures(text)
        docs.append((txt, val))
    except:
        print("Failed to add text: ", text)

print("Reading data...")
eb = pd.read_csv('emobank.csv', index_col=0)
nrc = pd.read_csv('NRC-VAD-Lexicon.txt', delim_whitespace=True, error_bad_lines=False)


#read emobank
for index, row in eb.iterrows():
    a = float(row["A"])
    text = row["text"]
    # change from 6 pt positive scale to range[-1, 1]
    a = a * 2 / 6 - 1
    addData(text, arousal_docs, a)
print("Building training sets...")
arousal_training_set = [({word: (True) for word in x[0]}, x[1]) for x in arousal_docs]

print("Training Arousal Classifier...")
arousal_classifier = NaiveBayesClassifier.train(arousal_training_set)

vader = SentimentIntensityAnalyzer()

print("Testing...")

def analyzeSentiment(sentence):
    #test_data_features = {word: (word in sentenceToFeatures(sentence)) for word in dictionary}
    test_data_features = {word: True for word in sentenceToFeatures(sentence)}
    vad = vader.polarity_scores(sentence)['compound']
    arousal = arousal_classifier.classify(test_data_features)
    return categorizeSentiment(vad, arousal)

# list of tuples (text, sentiment, phonemes)
annotated_text = []
for sentence in testData:
    sentiment = analyzeSentiment(sentence)
    phonemeCount = countPhonemes(sentence)
    annotated_text.append((sentence, sentiment, phonemeCount))
for entry in annotated_text:
    print(entry)
