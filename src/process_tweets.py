#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import json
from math import ceil, floor
from emoji import UNICODE_EMOJI

# noinspection PyCompatibility
from builtins import range

NUM_TRAIN = 90000
NUM_TEST = 10000
BALANCED = True

FASTTEXT_FILE = "../data/cc.nl.300.vec" 
SARCASTIC_TWEETS_FILE = "../data/sarcastic.json" 
NORMAL_TWEETS_FILE = "../data/non-sarcastic.json" 

# TODO: CROSS VALIDATION

def build_data():
    """
    Loads data
    """
    revs = []
    vocab = defaultdict(float)
    print("train tweets:" + str(NUM_TRAIN))
    print("test tweets:" + str(NUM_TEST))
    revs, vocab = get_revs('sarcastic', revs, vocab)
    revs, vocab = get_revs('normal', revs, vocab)

    return revs, vocab

def get_revs(type, revs, vocab):
    if type == 'sarcastic':
        tweets_file = SARCASTIC_TWEETS_FILE
        train_lines = floor(NUM_TRAIN * 0.5) if BALANCED else floor(NUM_TRAIN * 0.25)
        test_lines = floor(NUM_TEST * 0.5) if BALANCED else floor(NUM_TEST * 0.25)
        print("lines sarcastic: " + str(train_lines + test_lines))
        label = [1, 0]
    else:
        tweets_file = NORMAL_TWEETS_FILE
        train_lines = ceil(NUM_TRAIN * 0.5) if BALANCED else ceil(NUM_TRAIN * 0.75)
        test_lines = ceil(NUM_TEST * 0.5) if BALANCED else ceil(NUM_TEST * 0.75)
        print("lines normal: " + str(train_lines + test_lines))
        label = [0, 1]

    lines = train_lines + test_lines

    with open(tweets_file, encoding='UTF-8') as f:
        json_object = json.load(f)
        for i, line in enumerate(json_object):
            if i < lines:
                rev = []
                rev.append(line['tweet'].strip())
                orig_rev = clean_str(" ".join(rev))
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                orig_rev = (orig_rev.split())[0:100]
                orig_rev = " ".join(orig_rev)
                split = int(1) if i < train_lines else int (0)
                datum = {"id": line['id'],
                         "text": orig_rev,
                         "author": line['username'],
                         "label": label,
                         "num_words": len(orig_rev.split()),
                         "split": split}
                revs.append(datum)
            else:
                break
    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def loadGloveModel(gloveFile, vocab):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        if word in vocab:
               model[word] = embedding

    print("Done.",len(model)," words loaded!")
    return model

def load_fasttext(fname, vocab):
    """
    Loads 300x1 word vecs from Fasttext
    """
    print("Loading FastText Model")
    f = open(fname,'r', encoding='UTF-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        if word in vocab:
               model[word] = embedding

    print("Done.", len(model), " words loaded!")
    return model

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    emojis = extract_emoji(string)
    #string = re.sub(r"#\w*", "", string) # remove hashtags
    string = re.sub(r"#(sarcasme|sarcastisch|ironisch|ironie|cynisch|cynisme|not|niet)", "", string) # remove scraping hashtags
    # string = re.sub(r"#(sarcasm|sarcastic|ironic|irony|cynical|cynicism|not)", "", string) # remove english scraping hashtags
    # remove hyperlinks, copied from https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
    string = re.sub(r"https?:\/\/.*[\r\n]*", "", string)
    string = re.sub(r"\@\w*", "", string) # remove mentions
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # clean characters
    string = re.sub(r"(\`\w+)", " \1 ", string) # spaces around `word
    string = re.sub(r"(\'\w+)", " \1 ", string) # spaces around 'word
    string = re.sub(r",", " , ", string) # spaces around ,
    string = re.sub(r"!", " ! ", string) # spaces around !
    string = re.sub(r"\(", " ( ", string) # spaces around (
    string = re.sub(r"\)", " ) ", string) # spaces around )
    string = re.sub(r"\?", " ? ", string) # spaces around ?
    string = re.sub(r"\s{2,}", " ", string) # whitespace to single space
    string = add_emoji(string, emojis)
    return string.strip().lower() # make lowercase

def extract_emoji(string):
    return ''.join(c for c in string if c in UNICODE_EMOJI)

def add_emoji(string, emojis):
    for emoji in emojis:
        string = string + ' ' + emoji
    return(string)

if __name__=="__main__":
    w2v_file = FASTTEXT_FILE
    print("loading data...")
    revs, vocab = build_data()
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    w2v = load_fasttext(w2v_file, vocab) 
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open("pickle.p", "wb"))
    print("dataset created!")

