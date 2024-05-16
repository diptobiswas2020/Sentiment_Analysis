import os
for dirname, _, filenames in os.walk('/D:/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from nltk import wsd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import warnings

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')
nlp = load('en_core_web_sm')

! cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet # temp fix for lookup error.

data_Train = pd.read_csv('/D:/input/wsd-data/Train.csv')
data_Train

data_Test = pd.read_csv('/D:/input/wsd-data/Test.csv')
data_Test

for i in range(len(data_Train['text'])):
    print(data_Train['text'][i])
    print('\n')
    
for i in range(len(data_Test['text'])):
    print(data_Test['text'][i])
    print('\n')
    
#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import pandas as pd
from textblob import TextBlob
from nltk.tokenize.toktok import ToktokTokenizer
import re
tokenizer = ToktokTokenizer()
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

label_0 = data_Train[data_Train["label"]==0].sample(n=5000)
label_1 = data_Train[data_Train["label"]==1].sample(n=5000)

label_0 = data_Test[data_Test["label"]==0].sample(n=5000)
label_1 = data_Test[data_Test["label"]==1].sample(n=5000)

train = pd.concat([label_1,label_0])
from sklearn.utils import shuffle
train = shuffle(train)

test = pd.concat([label_1,label_0])
from sklearn.utils import shuffle
test = shuffle(test)

train.isnull().sum()
test.isnull().sum()

import numpy as np
train.replace(r'^\s*$', np.nan, regex=True,inplace=True)
train.dropna(axis = 0, how = "any", inplace = True)

import numpy as np
test.replace(r'^\s*$', np.nan, regex=True,inplace=True)
test.dropna(axis = 0, how = "any", inplace = True)

#removing escape seq
train.replace(to_replace = [r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

#removing escape seq
test.replace(to_replace = [r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

import numpy as np
train.replace(r'^\s*$', np.nan, regex=True,inplace=True)
train.dropna(axis = 0, how = "any", inplace = True)

import numpy as np
test.replace(r'^\s*$', np.nan, regex=True,inplace=True)
test.dropna(axis = 0, how = "any", inplace = True)

#removing non-ascii data
train["text"] = train["text"].str.encode("ascii", "ignore").str.decode("ascii")

#removing non-ascii data
test["text"] = test["text"].str.encode("ascii", "ignore").str.decode("ascii")

import string
string.punctuation

def remove_punctuations(text):
    import string
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    return text
train["text"]=train["text"].apply(remove_punctuations)
def remove_punctuations(text):
    import string
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    return text
test["text"]=test["text"].apply(remove_punctuations)

import nltk
from nltk.corpus import stopwords
print(stopwords.words("english"))
#removing "no" and "not" from stopwords
stopword_list = nltk.corpus.stopwords.words("english")
stopword_list.remove("no")
stopword_list.remove("not")

def custom_remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = " ".join(filtered_tokens)    
    return filtered_text

#applying stopwords remover function
train['text'] = train['text'].apply(custom_remove_stopwords)
#applying stopwords remover function
test['text'] = test['text'].apply(custom_remove_stopwords)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', " ", text)
    return text

train["text"] = train["text"].apply(remove_special_characters)
test["text"] = test["text"].apply(remove_special_characters)

def remove_html(text):
    import re
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r" ", text)

train["text"] = train["text"].apply(remove_html)
test["text"] = test["text"].apply(remove_html)

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r" ",text)

train["text"] = train["text"].apply(remove_URL)
test["text"] = test["text"].apply(remove_URL)

def remove_numbers(text):
    """ Removes integers """
    text = "".join([i for i in text if not i.isdigit()])         
    return text

train["text"] = train["text"].apply(remove_numbers)
test["text"] = test["text"].apply(remove_numbers)

def cleanse(word):
    rx = re.compile(r'\D*\d')
    if rx.match(word):
        return " "
    return word
def remove_alphanumeric(strings):
    nstrings = [" ".join(filter(None, (
    cleanse(word) for word in string.split()))) 
    for string in strings.split()]
    str1 = " ".join(nstrings)
    return str1

train["text"] = train["text"].apply(remove_alphanumeric)
test["text"] = test["text"].apply(remove_alphanumeric)

def lemmatize_text(text):
    text = nlp(text)
    text = " ".join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

train["text"] = train["text"].apply(lemmatize_text)
test["text"] = test["text"].apply(lemmatize_text)

for i in range(len(data_Train['text'])):
    text = data_Train['text'][i]
    sentences = sent_tokenize(text)
    print(sentences[0])
    print(sentences)
    print(len(sentences))
    print("\n")
    
for i in range(len(data_Train['text'])):
    text = data_Train['text'][i]
    sentences = sent_tokenize(text)
    print(len(sentences))
    for j in range(len(sentences)):
        sent = sentences[j]
        doc = nlp(sent)
        pos = []
        lemma = []
        text = []
        for tok in doc:
            wn.synsets(tok)
            k = 0
            for syn in wn.synsets(tok, pos=wn.NOUN):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.ADJ):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.VERB):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.NOUNC):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.ADJC):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.VERBC):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.VERBC):
                print(wsd.lesk(X.split(), tok))
                print(wsd.lesk(X.split(), tok).definition())
                k+=1
            pos.append(tok.pos_)
            lemma.append(tok.lemma_)
            text.append(tok.text)

for i in range(len(data_Train['text'])):
    text = data_Train['text'][i]
    sentences = sent_tokenize(text)
    print(len(sentences))
    for j in range(len(sentences)):
        sent = sentences[j]
        doc = nlp(sent)
        pos = []
        lemma = []
        text = []
        for tok in doc:
            pos.append(tok.pos_)
            lemma.append(tok.lemma_)
            text.append(tok.text)
        nlp_table = pd.DataFrame({'text':text,'lemma':lemma,'pos':pos})
        nlp_table.head()
        
        
for i in range(len(data_Test['text'])):
    text = data_Test['text'][i]
    sentences = sent_tokenize(text)
    print(sentences[0])
    print(sentences)
    print(len(sentences))
    print("\n")
    
for i in range(len(data_Test['text'])):
    text = data_Test['text'][i]
    sentences = sent_tokenize(text)
    print(len(sentences))
    for j in range(len(sentences)):
        sent = sentences[j]
        doc = nlp(sent)
        pos = []
        lemma = []
        text = []
        for tok in doc:
            wn.synsets(tok)
            k = 0
            for syn in wn.synsets(tok, pos=wn.NOUN):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.ADJ):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.VERB):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.NOUNC):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.ADJC):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.VERBC):
                print("defination {0} : {1}".format(k, syn.definition()))
                k+=1
            for syn in wn.synsets(tok, pos=wn.VERBC):
                print(wsd.lesk(X.split(), tok))
                print(wsd.lesk(X.split(), tok).definition())
                k+=1
            pos.append(tok.pos_)
            lemma.append(tok.lemma_)
            text.append(tok.text)

for i in range(len(data_Test['text'])):
    text = data_Test['text'][i]
    sentences = sent_tokenize(text)
    print(len(sentences))
    for j in range(len(sentences)):
        sent = sentences[j]
        doc = nlp(sent)
        pos = []
        lemma = []
        text = []
        for tok in doc:
            pos.append(tok.pos_)
            lemma.append(tok.lemma_)
            text.append(tok.text)
        nlp_table = pd.DataFrame({'text':text,'lemma':lemma,'pos':pos})
        nlp_table.head()
        

nlp_table.shape
nlp_table