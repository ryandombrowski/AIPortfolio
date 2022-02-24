import os
import sys
import json
import pandas as pd
import tokenize
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

wnl = WordNetLemmatizer()
stemmer = PorterStemmer()

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


# summaries = entries in the column labeled Summary
# put entries in summary column into an array
# Opening JSON file

df = pd.read_json('Musical_Instruments_5.json', lines=True)
f = open('Musical_Instruments_5.json')

# returns JSON object as
# a dictionary
print(df)
# Iterating through the json
# list

summaries = df.loc[:, 'summary']

summaries = summaries.str.replace('[^A-z ]', '').str.replace(' +', ' ').str.strip()
print(summaries)

# Read the document line by line

tokenized = []      # create a list for tokenized words
stem_sentence = []      # create a list for stemmed words
lemmatized = []     # create a list for lemmatized words

# Tokenization, Stemming, and Lemmatization
tokenized = [nltk.word_tokenize(str(word)) for word in summaries]   # Retrieving tokenized words
stem_sentence = [stemmer.stem(str(word)) for word in summaries]   # Retrieving stemmed words
lemmatized = [wnl.lemmatize(str(word)) for word in summaries]   # Retrieving lemmatized words


with open("outputtoken.txt", "w") as external_file:
    add_text = tokenized
    print(add_text, file=external_file)
    external_file.close()

with open("outputstem.txt", "w") as external_file:
    add_text = stem_sentence
    print(add_text, file=external_file)
    external_file.close()

with open("outputlemmatized.txt", "w") as external_file:
    add_text = lemmatized
    print(add_text, file=external_file)
    external_file.close()

print(tokenized)            # Output the tokenized words (print on screen or write to a file)
print(stem_sentence)        # Output the stemmed words (print on screen or write to a file)
print(lemmatized)           # Output the lemmatized words (print on screen or write to a file)

