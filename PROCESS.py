import pandas as pd
import numpy as np

import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import googletrans
import json
from mtranslate import translate

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet

nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
from stopwords import get_stopwords
from collections import Counter

def translates(comments):
    comment_engs = pd.Series()
    for idx, comment in enumerate(comments):
        english_translation = translate(comment, 'en')
        comment_engs[idx] = english_translation
    return comment_engs

# Sentiment Comments

def process_tweet_SC(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

def freq_sentence_SC(keys, vocab_dict):
    data_freq = []
    for word in keys:
        pos = 0
        neg = 0
        if (word, 1) in vocab_dict:
            pos = vocab_dict[(word, 1)]
        if (word, 0) in vocab_dict:
            neg = vocab_dict[(word, 0)]
        data_freq.append([word, pos, neg])
    return data_freq

def create_data_SC(comments, vocab):
    df_ = pd.DataFrame(columns=['idx', 'bias', 'positive', 'negative'])
    for idx, comment in enumerate(comments):
        list_freq = freq_sentence_SC(process_tweet_SC(comment), vocab_dict=vocab)
        positive_ = 0
        negative_ = 0
        for i in list_freq:
            positive_ += i[1]
            negative_ += i[2]
        new_row = {'idx': idx, 'bias': 1, 'positive': positive_, 'negative': negative_}
        df_.loc[len(df_)] = new_row
    return df_

def build_vocab_SC(tweets):

    all_positive_tweets = tweets.loc[tweets['sentiment'] == 1]['comment_eng']
    all_negative_tweets = tweets.loc[tweets['sentiment'] == 0]['comment_eng']

    all_positive_tweets = all_positive_tweets.to_list()
    all_negative_tweets = all_negative_tweets.to_list()

    ys = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))
    tweets = all_positive_tweets + all_negative_tweets
    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet_SC(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs, all_positive_tweets, all_negative_tweets

def create_data_train(all_positive_tweets, all_negative_tweets):
    df_ = pd.DataFrame(columns=['idx', 'bias', 'positive', 'negative', 'sentiment'])
    for idx, tweet in enumerate(all_positive_tweets):
        list_freq = freq_sentence_SC(process_tweet_SC(tweet))
        positive_ = 0
        negative_ = 0
        for i in list_freq:
            positive_ += i[1]
            negative_ += i[2]
        new_row = {'idx': idx, 'bias': 1, 'positive': positive_, 'negative': negative_, 'sentiment': 1}
        df_.loc[len(df_)] = new_row
    for idx, tweet in enumerate(all_negative_tweets):
        list_freq = freq_sentence_SC(process_tweet_SC(tweet))
        positive_ = 0
        negative_ = 0
        for i in list_freq:
            positive_ += i[1]
            negative_ += i[2]
        new_row = {'idx': idx, 'bias': 1, 'positive': positive_, 'negative': negative_, 'sentiment': 0}
        df_.loc[len(df_)] = new_row
    return df_

# Sentiment Split Comments

def process_tweet_SS(tweet):
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = nltk.pos_tag(tokenizer.tokenize(tweet))

    tweets_clean = []
    for word in tweet_tokens:
        if word[0] not in string.punctuation:
            stem_word = lemmatizer.lemmatize(word[0], pos_tag_convert(word[1]))
            tweets_clean.append(stem_word)

    return tweets_clean


def pos_tag_convert(nltk_tag: str) -> str:
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def padded_sequence(tweet, vocab_dict, max_len, unk_token='[UNK]'):
    unk_ID = vocab_dict[unk_token]
    tensor_l = []

    for word in tweet:
        word_ID = vocab_dict[word] if word in vocab_dict else unk_ID
        tensor_l.append(word_ID)
    padded_tensor = tensor_l + [0] * (max_len - len(tensor_l))

    return padded_tensor

def get_prediction_from_tweet(tweet, model, vocab, max_len):
    tweet = process_tweet_SS(tweet)
    tweet = padded_sequence(tweet, vocab, max_len)
    tweet = np.array([tweet])

    prediction = model.predict(tweet, verbose=False)

    return prediction[0][0]

def build_vocab_SS(train_x):

    vocab = {'': 0, '[UNK]': 1}

    for tweet in train_x:
        for word in tweet:
            if word not in vocab:
                vocab[word] = len(vocab)

    max_len = -1
    for tweet in train_x:
        if max_len < len(tweet): max_len = len(tweet)

    return vocab, max_len

# Split Comments

def preprocess_comments(comments):
    columns = ['idx', 'comment']
    comment_tachs = pd.DataFrame(columns=columns)

    for idx, comment in enumerate(comments):
        if ('\n\n' in comment): comment = comment.split('\n\n')[1]
        sentences = re.split(r'[,.!?;]\s*|\n', comment)
        sentences = [s for s in sentences if s]

        for i in sentences:
            new_row = {'idx': idx, 'comment': i}
            comment_tachs.loc[len(comment_tachs)] = new_row

    return comment_tachs

# Keyword

def generate_ngrams(tokens, n=2):
    return list(ngrams(tokens, n))

def keyword_extract(comments):

    df_keyword = pd.DataFrame(columns=['id', 'keyword', 'freq'])

    for idx, comment in comments.items():
        processed_comments = []
        processed_comments.append(process_tweet_SC(comments[idx]))

        bigrams = [generate_ngrams(comment, 2) for comment in processed_comments]
        trigrams = [generate_ngrams(comment, 3) for comment in processed_comments]

        bigrams = [item for sublist in bigrams for item in sublist]
        trigrams = [item for sublist in trigrams for item in sublist]

        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)

        keyword_b = ''
        for bigram, freq in bigram_freq.most_common(len(processed_comments[0])):
            keyword_b = ' '.join(bigram)
            new_row = {'id': idx, 'keyword': keyword_b, 'freq': freq}
            df_keyword.loc[len(df_keyword)] = new_row

        keyword_b = ''
        for trigram, freq in trigram_freq.most_common(len(processed_comments[0])):
            keyword_b = ' '.join(trigram)
            new_row = {'id': idx, 'keyword': keyword_b, 'freq': freq}
            df_keyword.loc[len(df_keyword)] = new_row

    df_keyword = df_keyword.groupby('keyword', as_index=False)['freq'].sum()
    df_keyword.sort_values(by='freq', ascending=False, inplace=True)

    return df_keyword

# Category Split Comments

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

def freq_sentence_CS(keys, label_check, vocab):
    data_freq = []

    for word in keys:
    
        dict_count = {}
        for col in label_check:
            dict_count[col] = 0
        
        for idx, i in enumerate(label_check):
            if (word, idx) in vocab:
                dict_count[i] = vocab[(word, idx)]
            
        list_temp = [word]
        list_temp.extend(dict_count.values())
        data_freq.append(list_temp)
    return data_freq

def create_data_CS(comments, label_check, vocab):
    temp = ['idx', 'bias'] + label_check
    df_ = pd.DataFrame(columns = temp)
    for idx, tweet in enumerate(comments):
        list_freq = freq_sentence_CS(process_tweet(tweet), label_check, vocab)

        dict_count = {}
        for col in label_check:
            dict_count[col] = 0
            
        for index, row in enumerate(list_freq):
            row = row[1:]
            for cate, value in zip(label_check, row):
                if(type(value) == str): continue
                dict_count[cate] += value

        new_row = { 'idx': idx, 'bias': 1 }
        for label in label_check:
            new_row[label] = dict_count[label]
        
        df_.loc[len(df_)] = new_row
    return df_