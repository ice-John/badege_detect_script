# coding=utf-8


import pandas as pd 
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
import nltk
nltk.download('punkt')
from nltk.corpus import wordnet as WN
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from bs4 import BeautifulSoup
import pickle
import os
import requests
import json
import sys

stop_words_en = set(stopwords.words('english'))

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def get_num_words(badge_response):
    text_to_process = badge_response

    tokens = word_tokenize(text_to_process)
    num_words = len(tokens) - 2

    return str(num_words)


def tokens(sent):
    return nltk.word_tokenize(sent)

def getNumSpellErrors(response):
    num_errors = 0

    for i in tokens(response):
        strip = i.rstrip()
        if not WN.synsets(strip):
            if strip in stop_words_en:    # <--- Check whether it's in stopword list
                #print("No mistakes :" + i)
                pass
            else:
                num_errors += 1
                #print("Wrong spellings : " +i)
        else:
            pass
            #print("No mistakes :" + i)
    return num_errors


def get_grammar(badge_response):
    api_endpoint = 'https://virtualwritingtutor.com/API/checkgrammar.php'
    api_key = 'ab4d4823-8b99-11e8-a062-00163e747eab'
    api_text = badge_response

    api_data = {
            'appKey':api_key,
            'text':api_text
            }

    r = requests.post(url = api_endpoint, data = api_data)
    data = json.loads(r.text)


    error_pct = data['error_grammar_percent']

    return error_pct




def main():
    tokenizer = pickle.load(open(SCRIPT_PATH + '/tokenizer.p','rb'))
    model = load_model( SCRIPT_PATH + '/model0.h5')
    message_input = sys.argv[1:]
    message_input[0] ='bid_'+str(message_input[0])
    message_input[1] ='sid_'+str(message_input[1])
    # print(message_input)
    X_tokens = tokenizer.texts_to_sequences([message_input])
    X_pad = pad_sequences(X_tokens, maxlen = 100, padding = 'post')
    #X_pad = X_pad.astype(np.float64)
    result_ls = list(model.predict_proba(X_pad))[0]
    #result_ls = result_ls.astype(np.float64)
#    num_words = get_num_words(message_input)
    nump_words = len(message_input) -2
#    NumSpellErrors = getNumSpellErrors(message_input[2:])
    #grammar = get_grammar(message)
    result = {'1':0,'2':0,'3':0,'4':0}
    i=0
    for k in result.keys():
    	result[k] = float(format(result_ls[i] * 100, '.2f'))
    	i=i+1
#    result['num_word']=num_words
#    result['NumSpellErrors']=NumSpellErrors
    #result['grammar']=grammar
    # print(result)
    print(json.dumps(result))



if __name__ == '__main__':
    main()