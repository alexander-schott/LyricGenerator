from nltk.classify import NaiveBayesClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer

# TestData: a list of sentences that will be annotated for sentiment and phoneme count.
testData = []
# The testing data is read out of the file 'test_dat.txt'
lyrics = open('test_data.txt', 'r')
for line in lyrics:
    testData.append(line)

"""
    Helper function to sentence tokenize a paragraph.
"""
def paragraphToSentecnes(paragraph):
    return nltk.sent_tokenize(paragraph)

"""
 Seperates a sentence into phonemes using cmudict lexicon, and counts the number of phonemes.
 Words not in the lexicon are assumed to be one phoneme.
"""
arpabet = nltk.corpus.cmudict.dict()
def countPhonemes(sentence):
    #split into words without punctuation
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
    print(count)
    return count

"""
    Removes stop words, and lemmatizes words in order to remove noise from data 
    and reduce the size of the number of values trained over.
    Takes in a sentence as a string, and returns a list of words.
"""
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

"""
    Returns an emotion value based on a valence and arousal score.
"""
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
# The emobank dataset contains text entries scored for valence, arousal, and dominance.
# Arousal values will be used to train the classifier
eb = pd.read_csv('emobank.csv', index_col=0)

# Data is read out of emobank and stored in a useful way in arousal_docs
arousal_docs = []


#read emobank
for index, row in eb.iterrows():
    a = float(row["A"])
    text = row["text"]
    # transform arousal value from 6 pt positive scale to range[-1, 1]
    a = a * 2 / 6 - 1
    try:
        # lemmatize, remove stop words, and transform sentence to list of words to build training docs
        txt = sentenceToFeatures(text)
        arousal_docs.append((txt, a))
    except:
        print("Failed to add text: ", text)

print("Building training sets...")
# Convert list of docs to dictionary format required by nltk classifier
arousal_training_set = [({word: (True) for word in x[0]}, x[1]) for x in arousal_docs]

print("Training Arousal Classifier...")
# Train classifier using naive bayes from nltk
arousal_classifier = NaiveBayesClassifier.train(arousal_training_set)

# Using nltk valence classifier, VADER. Trained on data from twitter.
vader = SentimentIntensityAnalyzer()

print("Testing...")

def analyzeSentiment(sentence):
    #convert string to format readable by classifier. Lemmatizes and removes stopwords as well.
    test_data_features = {word: True for word in sentenceToFeatures(sentence)}
    # the valence score using the VADER classifier
    valence = vader.polarity_scores(sentence)['compound']
    # arousal comes from the model trained in this program
    arousal = arousal_classifier.classify(test_data_features)
    return categorizeSentiment(valence, arousal)


# annotated text: list of tuples (text, sentiment, phonemes)
annotated_text = []

"""
    Classify sentiment for all sentences in testData, as well as count the phonemes.
    Result is stored in tuple in annotated_text
"""
for sentence in testData:
    sentiment = analyzeSentiment(sentence)
    phonemeCount = countPhonemes(sentence)
    annotated_text.append((sentence, sentiment, phonemeCount))

# printing annotated result to console.
for entry in annotated_text:
    print(entry)
