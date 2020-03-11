# coding=utf-8
import pickle
import os
import requests
import json
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import nltk
from nltk.corpus import wordnet as WN
from nltk import word_tokenize
from nltk.corpus import stopwords


GRAMMAR_API_ENDPOINT = 'https://virtualwritingtutor.com/API/checkgrammar.php'
GRAMMAR_API_KEY = 'ab4d4823-8b99-11e8-a062-00163e747eab'
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
STOP_WORDS_EN = set(stopwords.words('english'))

def get_num_words(text_to_process):
    tokens = word_tokenize(text_to_process)
    return int(len(tokens))

def tokens(sent):
    return nltk.word_tokenize(sent)

def num_spelling_errors(response):
    num_errors = 0

    for i in tokens(response):
        strip = i.rstrip()
        if not WN.synsets(strip):
            if strip in STOP_WORDS_EN:    # <--- Check whether it's in stopword list
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
    try:
        r = requests.post(url=GRAMMAR_API_ENDPOINT, data={
                          'appKey': GRAMMAR_API_KEY, 'text': badge_response})

        data = json.loads(r.text)
        if data['status'] == '401':
            raise Exception(data['message'])
        if 'error_grammar_percent' in data:
            return data['error_grammar_percent']
        else:
            raise Exception(r.text)
    except Exception as e:
        return {'error': repr(e)}
    else:
        return False


def get_scores(text):
    tokenizer = pickle.load(open(SCRIPT_PATH + '/tokenizer.p', 'rb'))
    model = load_model(SCRIPT_PATH + '/model0.h5')

    X_tokens = tokenizer.texts_to_sequences([text])
    X_pad = pad_sequences(X_tokens, maxlen=100, padding='post')
    result_ls = list(model.predict_proba(X_pad))[0]

    score_likeliness = {'1': 0, '2': 0, '3': 0, '4': 0,
                        '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    i = 0
    for k in score_likeliness.keys():
        score_likeliness[k] = float(format(result_ls[i] * 100, '.2f'))
        i = i + 1
    return score_likeliness


def main():
    text_input = sys.argv[1]
    result = {
        'scores': get_scores(text_input),
        'num_word': get_num_words(text_input),
        'num_spelling_errors': num_spelling_errors(text_input),
        'grammar': get_grammar(text_input)
    }
    print(json.dumps(result))

if __name__ == '__main__':
    main()