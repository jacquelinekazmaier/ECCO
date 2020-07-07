'''Contains functions for building, training and testing sentiment classification models'''

import sklearn
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import os, webbrowser, subprocess, threading, sys, multiprocessing
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from pattern.en import sentiment, wordnet
from nltk.corpus import opinion_lexicon
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, make_scorer, f1_score, \
    precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier

import keras
from keras import regularizers
from keras.callbacks import TensorBoard
import keras.backend as K

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import joblib
import gensim.downloader as api

from PyQt5.QtWidgets import QMessageBox

twitter = None
wiki = None

################################################
# lexicon-based methods
################################################
class Pattern_sentiment(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(np.unique(y)) == 2:
            self.binary = True
        else:
            self.binary = False
        return self

    def _predict(self, X, binary = None):
        text = X
        if self.binary == None:
            if binary == None:
                self.binary = False
            else:
                self.binary = binary

        predicted_sentiment = []
        for i in range(len(text)):
            if self.binary == False:
                if sentiment(text[i])[0] < 0:
                    predicted_sentiment.append(0)
                elif sentiment(text[i])[0] > 0.1:  # as per documentation
                    predicted_sentiment.append(2)
                else:
                    predicted_sentiment.append(1)
            else:
                if sentiment(text[i])[0] > 0.1:  # as per documentation
                    predicted_sentiment.append(1)
                else:
                    predicted_sentiment.append(0)
        return self.classes_[(np.array(predicted_sentiment).astype(int))]

    def predict(self, X, binary = None):
        if (str(self.classes_.dtype)[:3] != 'int'):
            return self._predict(X, binary)
        else:
            text = X
            if self.binary == None:
                if binary == None:
                    self.binary = False
                else:
                    self.binary = binary

            predicted_sentiment = []
            for i in range(len(text)):
                if self.binary == False:
                    if sentiment(text[i])[0] < 0:
                        predicted_sentiment.append(0)
                    elif sentiment(text[i])[0] > 0.1:  # as per documentation
                        predicted_sentiment.append(2)
                    else:
                        predicted_sentiment.append(1)
                else:
                    if sentiment(text[i])[0] > 0.1:  # as per documentation
                        predicted_sentiment.append(1)
                    else:
                        predicted_sentiment.append(0)
            return np.array(predicted_sentiment).astype(int)

    def predict_proba(self, X, binary=None):
        if self.binary == False:
            if (str(self.classes_.dtype)[:3] != 'int'):
                return label_binarize(self._predict(X, binary), self.classes_)
            else:
                return label_binarize(self.predict(X, binary), self.classes_)
        else:
            if (str(self.classes_.dtype)[:3] != 'int'):
                label = label_binarize(self._predict(X, binary), self.classes_)
                return np.hstack((1- label, label))

            else:
                label = label_binarize(self.predict(X, binary), self.classes_)
                return np.hstack((1- label, label))


class Sentiwordnet(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(np.unique(y)) == 2:
            self.binary = True
        else:
            self.binary = False
        return self

    def _predict(self, X, binary = None):
        tokenised_reviews = [review.split(" ") for review in X]

        if self.binary == None:
            if binary == None:
                self.binary = False
            else:
                self.binary = binary

        'implement lexicon-based sentiment model with SentiWordNet (rudimentary approach)'
        predicted_sentiment = []

        for review in tokenised_reviews:
            score = 0
            for token in review:
                wordscore = wordnet.sentiwordnet[token]
                if wordscore != None:
                    score += wordscore[0]
            if self.binary == True:
                if score > 0.1: # as per Pattern documentation
                    predicted_sentiment.append(1)
                else:
                    predicted_sentiment.append(0)
            else:
                if score > 0.1: # as per Pattern documentation
                    predicted_sentiment.append(2)
                elif score < 0:
                    predicted_sentiment.append(0)
                else:
                    predicted_sentiment.append(1)

        return self.classes_[(np.array(predicted_sentiment).astype(int))]

    def predict(self, X, binary=None):
        if (str(self.classes_.dtype)[:3] != 'int'):
            return self._predict(X, binary)
        else:
            tokenised_reviews = [review.split(" ") for review in X]

            if self.binary == None:
                if binary == None:
                    self.binary = False
                else:
                    self.binary = binary

            'implement lexicon-based sentiment model with SentiWordNet (rudimentary approach)'
            predicted_sentiment = []

            for review in tokenised_reviews:
                score = 0
                for token in review:
                    wordscore = wordnet.sentiwordnet[token]
                    if wordscore != None:
                        score += wordscore[0]
                if self.binary == True:
                    if score > 0.1:  # as per Pattern documentation
                        predicted_sentiment.append(1)
                    else:
                        predicted_sentiment.append(0)
                else:
                    if score > 0.1:  # as per Pattern documentation
                        predicted_sentiment.append(2)
                    elif score < 0:
                        predicted_sentiment.append(0)
                    else:
                        predicted_sentiment.append(1)
        return np.array(predicted_sentiment).astype(int)

    def predict_proba(self, X, binary=None):
        if self.binary == False:
            if (str(self.classes_.dtype)[:3] != 'int'):
                return label_binarize(self._predict(X, binary), self.classes_)
            else:
                return label_binarize(self.predict(X, binary), self.classes_)
        else:
            if (str(self.classes_.dtype)[:3] != 'int'):
                label = label_binarize(self._predict(X, binary), self.classes_)
                return np.hstack((1 - label, label))

            else:
                label = label_binarize(self.predict(X, binary), self.classes_)
                return np.hstack((1 - label, label))


class Hu_liu_sentiment(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(np.unique(y)) == 2:
            self.binary = True
        else:
            self.binary = False
        return self

    def _predict(self, X, binary=None):
        tokenised_reviews = [review.split(" ") for review in X]

        if self.binary == None:
            if binary == None:
                self.binary = False
            else:
                self.binary = binary

        'implement sentiment analyser using lexicon from Hu and Liu'
        predicted_sentiment = []

        for review in tokenised_reviews:
            pos_words = sum(token in review for token in opinion_lexicon.positive())
            neg_words = sum(token in review for token in opinion_lexicon.negative())

            if self.binary == True:
                if pos_words > neg_words:
                    predicted_sentiment.append(1)
                else:
                    predicted_sentiment.append(0)
            else:
                if pos_words > neg_words:
                    predicted_sentiment.append(2)
                elif pos_words < neg_words:
                    predicted_sentiment.append(0)
                else:
                    predicted_sentiment.append(1)

        return self.classes_[(np.array(predicted_sentiment).astype(int))]

    def predict(self, X, binary=None):
        if (str(self.classes_.dtype)[:3] != 'int'):
            return self._predict(X, binary)
        else:
            tokenised_reviews = [review.split(" ") for review in X]

            if self.binary == None:
                if binary == None:
                    self.binary = False
                else:
                    self.binary = binary

            'implement sentiment analyser using lexicon from Hu and Liu'
            predicted_sentiment = []

            for review in tokenised_reviews:
                pos_words = sum(token in review for token in opinion_lexicon.positive())
                neg_words = sum(token in review for token in opinion_lexicon.negative())

                if self.binary == True:
                    if pos_words > neg_words:
                        predicted_sentiment.append(1)
                    else:
                        predicted_sentiment.append(0)
                else:
                    if pos_words > neg_words:
                        predicted_sentiment.append(2)
                    elif pos_words < neg_words:
                        predicted_sentiment.append(0)
                    else:
                        predicted_sentiment.append(1)

            return np.array(predicted_sentiment).astype(int)

    def predict_proba(self, X, binary=None):
        if self.binary == False:
            if (str(self.classes_.dtype)[:3] != 'int'):
                return label_binarize(self._predict(X, binary), self.classes_)
            else:
                return label_binarize(self.predict(X, binary), self.classes_)
        else:
            if (str(self.classes_.dtype)[:3] != 'int'):
                label = label_binarize(self._predict(X, binary), self.classes_)
                return np.hstack((1 - label, label))

            else:
                label = label_binarize(self.predict(X, binary), self.classes_)
                return np.hstack((1 - label, label))


class Vader(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.binary = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(np.unique(y)) == 2:
            self.binary = True
        else:
            self.binary = False
        return self

    def _predict(self, X, binary=None):
        text = X
        analyzer = SentimentIntensityAnalyzer()
        predicted_sentiment = []

        if binary != None:
            self.binary = binary

        if self.binary == True:
            for i in range(len(text)):
                sentiment = analyzer.polarity_scores(text[i])
                if sentiment['compound'] >= 0.05:
                    predicted_sentiment.append(int(np.where(self.classes_=='positive')[0]))
                else:
                    predicted_sentiment.append(int(np.where(self.classes_=='negative')[0]))

        else:
            for i in range(len(text)):
                sentiment = analyzer.polarity_scores(text[i])
                if sentiment['compound'] <= -0.05:
                    predicted_sentiment.append(int(np.where(self.classes_=='negative')[0]))
                elif sentiment['compound'] >= 0.05:  # as per documentation???
                    predicted_sentiment.append(int(np.where(self.classes_=='positive')[0]))
                else:
                    predicted_sentiment.append(int(np.where(self.classes_=='neutral')[0]))

        return self.classes_[(np.array(predicted_sentiment).astype(int))]

    def predict(self, X, binary=None):
        if(str(self.classes_.dtype)[:3] != 'int'):
            return self._predict(X, binary)
        else:
            text = X
            analyzer = SentimentIntensityAnalyzer()
            predicted_sentiment = []

            if binary != None:
                self.binary = binary

            if self.binary == True:
                for i in range(len(text)):
                    sentiment = analyzer.polarity_scores(text[i])
                    if sentiment['compound'] >= 0.05:
                        predicted_sentiment.append(1)
                    else:
                        predicted_sentiment.append(0)

            else:
                for i in range(len(text)):
                    sentiment = analyzer.polarity_scores(text[i])
                    if sentiment['compound'] <= -0.05:
                        predicted_sentiment.append(0)
                    elif sentiment['compound'] >= 0.05:  # as per documentation???
                        predicted_sentiment.append(2)
                    else:
                        predicted_sentiment.append(1)

        return np.array(predicted_sentiment).astype(int)

    def predict_proba(self, X, binary=None):
        if self.binary == False:
            if (str(self.classes_.dtype)[:3] != 'int'):
                return label_binarize(self._predict(X, binary), self.classes_)
            else:
                return label_binarize(self.predict(X, binary), self.classes_)
        else:
            if (str(self.classes_.dtype)[:3] != 'int'):
                label = label_binarize(self._predict(X, binary), self.classes_)
                return np.hstack((1 - label, label))

            else:
                label = label_binarize(self.predict(X, binary), self.classes_)
                return np.hstack((1 - label, label))



################################################
# Helper functions
################################################
def vectorise(type, max_features, n_grams=(1, 1)):
    if type == 'presence':
        vectoriser = sklearn.feature_extraction.text.CountVectorizer(binary=True, lowercase=False,
                                                                     ngram_range=n_grams, max_features=max_features)
    elif type == 'frequency':
        vectoriser = sklearn.feature_extraction.text.CountVectorizer(binary=False, lowercase=False,
                                                                     ngram_range=n_grams, max_features=max_features)
    elif type == 'tf-idf':
        vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(binary=False, lowercase=False,
                                                                     ngram_range=n_grams, max_features=max_features)
    return vectoriser

def vec_split(reviews, type, n_grams=(1, 1), testsize=0.2, randomstate=101, sample_type='stratify', max_features=None):
    # create vectoriser
    if type == 'presence':
        vectoriser = sklearn.feature_extraction.text.CountVectorizer(binary=True, lowercase=False,
                                                                     ngram_range=n_grams, max_features=max_features)
    elif type == 'frequency':
        vectoriser = sklearn.feature_extraction.text.CountVectorizer(binary=False, lowercase=False,
                                                                     ngram_range=n_grams, max_features=max_features)
    elif type == 'tf-idf':
        vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(binary=False, lowercase=False,
                                                                     ngram_range=n_grams, max_features=max_features)

    if testsize == 0:
        # don't split just vectorise
        return vectoriser.fit_transform(reviews.cleantext).toarray(), reviews.labels
    else:
        # split into training and test data
        idx_train, idx_test, y_train, y_test = train_test_split(reviews.labelled_data.index, reviews.labels,
                                                                test_size=testsize,
                                                                random_state=randomstate, stratify=reviews.labels)

        # fit vocabulary from training data, return term-document-matrix of train and test data using this vocab
        X_train = vectoriser.fit_transform(reviews.cleantext[idx_train]).toarray()
        X_test = vectoriser.transform(reviews.cleantext[idx_test]).toarray()

        return X_train, X_test, idx_train, idx_test, y_train, y_test

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

class LaunchTensorboard(multiprocessing.Process):
    def __init__(self, model):
        super(LaunchTensorboard, self).__init__()
        self.model = model

    def run(self):
        if self.model == 'ANN':
            threading.Timer(2, lambda: webbrowser.open('127.0.0.1:8081')).start()
            subprocess.call(['tensorboard', '--logdir=logs/ANN', '--host=127.0.0.1',
                             '--port=8081'])
        elif self.model == 'CNN':
            threading.Timer(2, lambda: webbrowser.open('127.0.0.1:8082')).start()
            subprocess.call(['tensorboard', '--logdir=logs/CNN', '--host=127.0.0.1',
                             '--port=8082'])
        else:
            threading.Timer(2, lambda: webbrowser.open('127.0.0.1:8083')).start()
            subprocess.call(['tensorboard', '--logdir=logs/LSTM', '--host=127.0.0.1',
                             '--port=8083'])

################################################
# Machine learning models
################################################
def train_NB(type):
    if type == 'presence':
        nb = BernoulliNB()
    elif type == 'frequency':
        nb = MultinomialNB()
    else:
        nb = GaussianNB()
        parameters = {}
    return nb

class ANN(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers = [100], activation_function = 'relu', loss_function = 'binary_crossentropy', reg_type = None, reg_param = 0, dropout_prob = 0, batchnorm = False,
            solver = 'Adam', max_epochs = 200, learning_rate = 0.2, decay = 0, val_split=0.2):
        self.model = keras.models.Sequential()
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.dropout_prob = dropout_prob
        self.batchnorm = batchnorm
        self.solver = solver
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.val_split = val_split

    def fit(self, X, y):
        #build model
        if self.reg_type == 'L1':
            reg = regularizers.l1(self.reg_param)
        elif self.reg_type == 'L2':
            reg = regularizers.l2(self.reg_param)
        else:
            reg = None

        for i in range(len(self.hidden_layers)):
            if i == 0: # input layer
                self.model.add(
                    keras.layers.Dense(self.hidden_layers[0], input_dim=X[0].shape[0],
                          kernel_regularizer=reg))
                if self.batchnorm:
                    self.model.add(
                        keras.layers.BatchNormalization(momentum=0.99))
                self.model.add(keras.layers.Activation(activation=self.activation_function))
            else:
                self.model.add(keras.layers.Dense(self.hidden_layers[0],
                                     kernel_regularizer=reg))
                if self.batchnorm:
                    self.model.add(
                        keras.layers.BatchNormalization(momentum=0.99))
                self.model.add(keras.layers.Activation(activation=self.activation_function))
            if self.dropout_prob > 0:
                self.model.add(keras.layers.Dropout(rate = self.dropout_prob))

        # output layer
        if len(np.unique(y)) > 2:
            self.model.add(keras.layers.Dense(len(np.unique(y)), activation='sigmoid'))
        else:
            self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        print(self.model.summary())

        #fit model
        self.classes_ = np.unique(y)
        y = label_binarize(y, self.classes_)

        if self.solver == 'sgd':
            solver = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=0, nesterov=False)
        if self.solver == 'momentum':
            solver = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=0.9, nesterov=False)
        if self.solver == 'Adam':
            solver = keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay, beta_1=0.9, beta_2=0.999)
        else: #if self.solver == 'RMSprop':
            solver = keras.optimizers.RMSprop(lr=self.learning_rate, decay=self.decay, rho = 0.9)

        self.model.compile(loss=self.loss_function, optimizer=solver, metrics=['accuracy'])
        logdir = os.path.join("logs", "ANN",datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model.fit(X, y, batch_size=32, epochs = self.max_epochs, verbose=2, validation_split=self.val_split,
                        callbacks=[TrainValTensorBoard(log_dir=logdir, write_graph=False)])

        return self

    def predict(self, X):

        scores = self.model.predict(X)

        if len(self.classes_) > 2:
            preds = self.classes_[np.argmax(scores, axis=1)]
            return preds
        else:
            preds = self.classes_[(scores > 0.5).astype(int)]
            return np.array([item[0] for item in preds])

    def predict_proba(self, X):

        return_val = self.model.predict(X)

        if len(self.classes_) > 2:
            return return_val
        else:
            return np.hstack((1 - return_val, return_val))

class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=[[1,1,1]], embedding_size=10, convolution_type = 'valid', max_feat = 500,
                 pooling = 'yes',
                 pooling_filter_size = 2, pooling_filter_stride = 2, batchnorm = False,
                activation_function='relu', loss_function='binary_crossentropy',
                 reg_type=None, reg_param=0,
                 solver='Adam', max_epochs=200, learning_rate=0.2, decay=0, val_split=0.2):

        self.model = keras.models.Sequential()
        self.hidden_layers = hidden_layers
        self.embedding_size = embedding_size
        self.convolution_type = convolution_type
        self.max_feat = max_feat
        self.pooling = pooling
        self.pooling_filter_size = pooling_filter_size
        self.pooling_filter_stride = pooling_filter_stride
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.solver = solver
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.val_split = val_split
        self.batchnorm = batchnorm

    def fit(self, X, y):
        global twitter, wiki
        #get word embeddings
        self.tokeniser = Tokenizer(num_words = self.max_feat)
        self.tokeniser.fit_on_texts(X)
        X_seq = self.tokeniser.texts_to_sequences(X)

        self.maxlen = max(map(len,X)) # set maxlen to max observed length in training data
        X_seq = pad_sequences(X_seq, padding='post', maxlen=self.maxlen)

        #create word embeddings
        if self.embedding_size == 'Twitter':
            print('Twitter')
            vocab_size = len(self.tokeniser.word_index) + 1  # +1 because of reserved 0 index
            if twitter == None:
                pre_trained = api.load('glove-twitter-25')
                twitter = pre_trained
            else:
                pre_trained = twitter
            word_index = self.tokeniser.word_index
            embedding_matrix = np.zeros((vocab_size, 25))
            for word, i in word_index.items():
                if word in pre_trained.vocab:
                    embedding_matrix[i] = pre_trained.word_vec(word)
            print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
            print('i.e. %d %' % round(np.sum(np.sum(embedding_matrix, axis=1) == 0)/embedding_matrix.shape[0]*100, 2))
            embedding_layer = keras.layers.Embedding(embedding_matrix.shape[0],  # or len(word_index) + 1
                                        embedding_matrix.shape[1],  # or EMBEDDING_DIM,
                                        weights=[embedding_matrix],
                                        input_length=self.maxlen,
                                        trainable=False)
            self.model.add(
                embedding_layer
            )
        elif self.embedding_size == 'Wiki':
            print('Wiki')
            vocab_size = len(self.tokeniser.word_index) + 1  # +1 because of reserved 0 index
            if wiki == None:
                pre_trained = api.load('glove-wiki-gigaword-100')
                wiki = pre_trained
            else:
                pre_trained = wiki
                word_index = self.tokeniser.word_index
                embedding_matrix = np.zeros((vocab_size, 25))
            for word, i in word_index.items():
                if word in pre_trained.vocab:
                    embedding_matrix[i] = pre_trained.word_vec(word)
            print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
            print('i.e. %d %' % round(np.sum(np.sum(embedding_matrix, axis=1) == 0) / embedding_matrix.shape[0] * 100, 2))
            embedding_layer = keras.layers.Embedding(embedding_matrix.shape[0],  # or len(word_index) + 1
                                                     embedding_matrix.shape[1],  # or EMBEDDING_DIM,
                                                     weights=[embedding_matrix],
                                                     input_length=self.maxlen,
                                                     trainable=False)
            self.model.add(
                embedding_layer
            )

        else:
            print('Own embeddings')
            vocab_size = self.max_feat + 1  # +1 because of reserved 0 index
            self.model.add(
                keras.layers.Embedding(input_dim=vocab_size,
                                       output_dim=self.embedding_size,
                                       input_length = self.maxlen)
            )

        #check regularisation params
        if self.reg_type == 'L1':
            reg = regularizers.l1(self.reg_param)
        elif self.reg_type == 'L2':
            reg = regularizers.l2(self.reg_param)
        else:
            reg = None

        #add convolutional layers
        for layer in self.hidden_layers:
            self.model.add(
                keras.layers.Conv1D(kernel_size = layer[0], filters = layer[2],
                                    strides = layer[1], padding = self.convolution_type,
                                   kernel_regularizer=reg)
            )
            if self.batchnorm == True:
                self.model.add(keras.layers.BatchNormalization(momentum=0.0001))
            self.model.add(keras.layers.Activation(activation=self.activation_function))
            if self.pooling == 'yes':
                self.model.add(
                    keras.layers.MaxPooling1D(pool_size=self.pooling_filter_size, strides=self.pooling_filter_stride)
                )

        #add output layer
        self.model.add(keras.layers.Flatten())
        # output layer
        if len(np.unique(y)) > 2:
            self.model.add(keras.layers.Dense(len(np.unique(y)), activation='sigmoid'))
        else:
            self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        print(self.model.summary())

        # fit model
        self.classes_ = np.unique(y)
        y = label_binarize(y, self.classes_)

        if self.solver == 'sgd':
            solver = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=0, nesterov=False)
        if self.solver == 'momentum':
            solver = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=0.9, nesterov=False)
        if self.solver == 'Adam':
            solver = keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay, beta_1=0.9, beta_2=0.999)
        else:  # if self.solver == 'RMSprop':
            solver = keras.optimizers.RMSprop(lr=self.learning_rate, decay=self.decay, rho=0.9)

        self.model.compile(loss=self.loss_function, optimizer=solver, metrics=['accuracy'])

        logdir = os.path.join("logs", "CNN", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model.fit(X_seq, y, batch_size=32, epochs=self.max_epochs, verbose=2, validation_split=self.val_split,
                       callbacks=[TrainValTensorBoard(log_dir=logdir, write_graph=False)])
        return self

    def predict(self, X):
        X_seq = self.tokeniser.texts_to_sequences(X)
        X_seq = pad_sequences(X_seq, padding='post', maxlen=self.maxlen)

        scores = self.model.predict(X_seq)

        if len(self.classes_) > 2:
            preds = self.classes_[np.argmax(scores, axis=1)]
            return preds
        else:
            preds = self.classes_[(scores > 0.5).astype(int)]
            return np.array([item[0] for item in preds])

    def predict_proba(self, X):
        X_seq = self.tokeniser.texts_to_sequences(X)
        X_seq = pad_sequences(X_seq, padding='post', maxlen=self.maxlen)

        return_val = self.model.predict(X_seq)

        if len(self.classes_) > 2:
            return return_val
        else:
            return np.hstack((1 - return_val, return_val))

class LSTM(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=[10], embedding_size=5, max_feat=500,
                 dropout_prob = 0,
                 loss_function='binary_crossentropy',
                 reg_type=None, reg_param=0,
                 solver='Adam', max_epochs=10, learning_rate=0.2, decay=0, val_split=0.2,
                 batch_size=125):

        self.model = keras.models.Sequential()
        self.hidden_layers = hidden_layers
        self.embedding_size = embedding_size
        self.max_feat = max_feat
        self.dropout_prob = dropout_prob
        self.loss_function = loss_function
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.solver = solver
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.val_split = val_split
        self.batch_size = batch_size

    def fit(self, X, y):

        #get word embeddings
        self.tokeniser = Tokenizer(num_words = self.max_feat)
        self.tokeniser.fit_on_texts(X)
        X_seq = self.tokeniser.texts_to_sequences(X)
        vocab_size = self.max_feat + 1  # +1 because of reserved 0 index

        self.maxlen = max(map(len,X)) # set maxlen to max observed length in training data
        X_seq = pad_sequences(X_seq, padding='post', maxlen=self.maxlen)

        #create word embeddings
        self.model.add(
            keras.layers.Embedding(input_dim=self.max_feat,
                                   output_dim=self.embedding_size,
                                   input_length = self.maxlen)
        )

        #check regularisation params
        if self.reg_type == 'L1':
            reg = regularizers.l1(self.reg_param)
        elif self.reg_type == 'L2':
            reg = regularizers.l2(self.reg_param)
        else:
            reg = None

        # create LSTM layer(s)
        for layer in self.hidden_layers:
            self.model.add(
                keras.layers.LSTM(layer, dropout = self.dropout_prob, recurrent_dropout=self.dropout_prob,
                                  return_sequences=True)
            )

        # add output layer
        self.model.add(keras.layers.Flatten())
        # output layer
        if len(np.unique(y)) > 2:
            self.model.add(keras.layers.Dense(len(np.unique(y)), activation='sigmoid'))
        else:
            self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        print(self.model.summary())

        # fit model
        self.classes_ = np.unique(y)
        y = label_binarize(y, self.classes_)

        if self.solver == 'sgd':
            solver = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=0, nesterov=False)
        if self.solver == 'momentum':
            solver = keras.optimizers.SGD(lr=self.learning_rate, decay=self.decay, momentum=0.9, nesterov=False)
        if self.solver == 'Adam':
            solver = keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay, beta_1=0.9, beta_2=0.999)
        else:  # if self.solver == 'RMSprop':
            solver = keras.optimizers.RMSprop(lr=self.learning_rate, decay=self.decay, rho=0.9)

        self.model.compile(loss=self.loss_function, optimizer=solver, metrics=['accuracy'])

        logdir = os.path.join("logs", "LSTM", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model.fit(X_seq, y, batch_size=self.batch_size, epochs=self.max_epochs, verbose=2, validation_split=self.val_split,
                       callbacks=[TrainValTensorBoard(log_dir=logdir, write_graph=False)])
        return self

    def predict(self, X):
        X_seq = self.tokeniser.texts_to_sequences(X)
        X_seq = pad_sequences(X_seq, padding='post', maxlen=self.maxlen)


        scores = self.model.predict(X_seq)

        if len(self.classes_) > 2:
            preds = self.classes_[np.argmax(scores, axis=1)]
            return preds
        else:
            preds = self.classes_[(scores > 0.5).astype(int)]
            return np.array([item[0] for item in preds])

    def predict_proba(self, X):
        X_seq = self.tokeniser.texts_to_sequences(X)
        X_seq = pad_sequences(X_seq, padding='post', maxlen=self.maxlen)

        return_val = self.model.predict(X_seq)
        if len(self.classes_) > 2:
            return return_val
        else:
            return np.hstack((1 - return_val, return_val))

def generate_ensemble(base_classifiers, configuration, accuracies = None):

    if configuration['combinationMethod'] == 'simple':
        if configuration['outputType'] == 'discrete':
            return VotingClassifier(base_classifiers)
        else:
            try:
                return VotingClassifier(base_classifiers, voting='soft')
            except:
                msgBox = QMessageBox()
                msgBox.setText("Probabilities not defined for one of the models. Discrete labels used instead.")
                msgBox.setWindowTitle("Error")
                msgBox.exec_()
                return VotingClassifier(base_classifiers)

    elif configuration['combinationMethod'] == 'weighted':
        weights = np.array(accuracies)/sum(np.array(accuracies))
        if configuration['outputType'] == 'discrete':
            return VotingClassifier(base_classifiers, weights=weights)
        else:
            try:
                return VotingClassifier(base_classifiers, weights=weights, voting='soft')
            except:
                msgBox = QMessageBox()
                msgBox.setText("Probabilities not defined for one of the models. Discrete labels used instead.")
                msgBox.setWindowTitle("Error")
                msgBox.exec_()
                return VotingClassifier(base_classifiers, weights=weights)

    elif configuration['combinationMethod'] == 'meta':
        if configuration['outputType'] == 'discrete':
            return StackingClassifier(base_classifiers,
                                      cv = 5,
                                      stack_method='predict')
        else: #soft labels
            try:
                return StackingClassifier(base_classifiers,
                                          cv = 5,
                                          stack_method='auto')
            except:
                msgBox = QMessageBox()
                msgBox.setText("Probabilities not defined for one of the models. Discrete labels used instead.")
                msgBox.setWindowTitle("Error")
                msgBox.exec_()
                return StackingClassifier(base_classifiers,
                                          cv=5,
                                          stack_method='predict')


################################################
# Model evaluation and selection
################################################
def plot_confusion_matrix(y_true, y_pred, classes, name, normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(dpi=800)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_title(title, color='white')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    im.axes.tick_params(color='white', labelcolor='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    cb.set_label('Number of observations', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(name, transparent=True, dpi=800)
    return None

def evaluate(true_labels, predicted_labels, path=None, return_type=None):
    classes = true_labels.unique()
    classes.sort()
    num_classes = len(classes)

    if predicted_labels['labels'].size == 1:
        labelpred = predicted_labels['labels'][0]
        numpred = predicted_labels['numerical'][0]
    else:
        labelpred = predicted_labels['labels']
        numpred = predicted_labels['numerical']

    def multi_roc(true, pred):
        """
        Multiclass ROC value

        :param true: vector of true labels of shape [n_samples]
        :param pred: matrix of predicted scores/probabilities for each label of shape [n_samples],[n_samples, n_classes]
               (output of predict_proba)
        :return: macro and micro-averaged AUC scores for a multi class problem
        """
        i = 0
        auc = {}
        macro_auc = 0
        micro_auc = 0

        for class_i in classes:
            y_i = true == class_i  # convert to just one class
            p_i = []
            for j in range(len(pred)):
                # extract numerical predictions for class i (necessary due to them not being in separate columns)
                p_i.append(pred[j][i])
            i += 1
            auc_i = roc_auc_score(y_i, p_i)
            auc[class_i] = auc_i
            macro_auc += auc_i
            micro_auc += auc_i * sum(y_i)

        macro_auc /= num_classes
        micro_auc /= len(true)  # num_samples

        print('ROC AUC value per class:')
        print(auc)
        print('macro avg\t', macro_auc)
        print('micro avg\t', micro_auc)
        return macro_auc, micro_auc

    warnings.filterwarnings('ignore')  # ignore warning if a class is not predicted
    print(classification_report(true_labels, labelpred))
    accuracy = accuracy_score(true_labels, labelpred)
    precision = precision_score(true_labels, labelpred, average='micro')
    recall = recall_score(true_labels, labelpred, average='micro')
    f1 = f1_score(true_labels, labelpred, average='micro')
    print('accuracy: ', accuracy)

    if path is not None:
        plot_confusion_matrix(true_labels, labelpred, classes, path)

    if np.any(numpred == None):
        # discrete classifier -> binarise output
        numpred = label_binarize(labelpred, classes=classes)  # same as predict_proba for discrete

    try:
        macro_auc, micro_auc = multi_roc(true_labels, numpred)
    except:
        macro_auc = roc_auc_score(true_labels, numpred)
        micro_auc = roc_auc_score(true_labels, numpred)
        print('Macro AUC: ', macro_auc)
        print('Micro AUC: ', micro_auc)

    if return_type == 'AUC':
        return micro_auc
    if return_type == 'all':
        return float(round(micro_auc, 3)), float(round(accuracy,3)), float(round(precision,3)), float(round(recall,3)), float(round(f1, 3))

def get_multi_roc(estimator, X, y):
    """Function for purpose of evaluating alternatives using AUC in grid search"""
    try:
        predictions = pd.DataFrame({'labels': [estimator.predict(X)], 'numerical': [estimator.predict_proba(X)]})
    except:
        predictions = pd.DataFrame({'labels': [estimator.predict(X)], 'numerical': None})
    auc = evaluate(y, predictions, return_type='AUC')
    print('auc:', auc)
    return auc

def grid_search(model, X_train, y_train, parameters, metric, nfolds=3):
    if metric == 'Accuracy':
        scorer_fct = make_scorer(accuracy_score)
    elif metric == 'F1 Score':
        scorer_fct = make_scorer(f1_score, average='micro')
    elif metric == 'Precision':
        scorer_fct = make_scorer(precision_score, average='micro')
    elif metric == 'Recall':
        scorer_fct = make_scorer(recall_score, average='micro')
    else: # metric == 'AUC':
        scorer_fct = get_multi_roc  # also micro average

    gridsearch = RandomizedSearchCV(model, n_iter=10, param_distributions=parameters, cv=nfolds,
                                                     return_train_score=False, n_jobs=8, verbose=2, scoring=scorer_fct,
                                    pre_dispatch = 16)
    K.clear_session()
    try:
        gridsearch.fit(X_train, y_train)
    except:
        with joblib.parallel_backend('threading'):
            gridsearch.fit(X_train, y_train)
    print('best params:\n')
    print(gridsearch.best_params_)
    print('best score: ', gridsearch.best_score_)

    return gridsearch.best_estimator_, gridsearch.best_score_, gridsearch.best_params_