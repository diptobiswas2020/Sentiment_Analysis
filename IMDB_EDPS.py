import os
for dirname, _, filenames in os.walk('/D:/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
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

train=pd.read_csv("/D:/input/senti-data/IMDB_Train.csv")
train

test=pd.read_csv("/D:/input/senti-data/IMDB_Test.csv")
test 

label_0 = train[train["label"]==0].sample(n=5000)
label_1 = train[train["label"]==1].sample(n=5000)

label_0 = test[test["label"]==0].sample(n=5000)
label_1 = test[test["label"]==1].sample(n=5000)

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

