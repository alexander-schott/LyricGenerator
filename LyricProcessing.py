from nltk.classify import NaiveBayesClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer

# a list of sentences
testData = []
lyrics = open('test_data.txt', 'r')
for line in lyrics:
    testData.append(line)


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

print("Reading data...")
eb = pd.read_csv('emobank.csv', index_col=0)


#read emobank
for index, row in eb.iterrows():
    a = float(row["A"])
    text = row["text"]
    # change from 6 pt positive scale to range[-1, 1]
    a = a * 2 / 6 - 1
    try:
        txt = sentenceToFeatures(text)
        arousal_docs.append((txt, a))
    except:
        print("Failed to add text: ", text)

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
