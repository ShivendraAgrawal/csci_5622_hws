import os
import json
from collections import Counter
from csv import DictReader, DictWriter
from pprint import pprint

import numpy as np
import time
from numpy import array
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

SEED = 50


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features

class NumSentenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 3))
        i = 0
        for ex in examples:
            features[i, 0] = ex.count('.')
            features[i, 1] = ex.count(',')
            features[i, 2] = ex.count('!')
            i += 1

        return features

class PatternTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        return [{'count_question': len(re.findall(r'\?', text)),
                'count_special_chars': len(re.findall(r'[^a-z0-9A-Z\w\.\,\?\!]', text)),
                 'count_num': len(re.findall(r'[0-9]+', text))}
         for text in examples]


# TODO: Add custom feature transformers for the movie review data


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),

            ('sentence_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('num_sentence', NumSentenceTransformer())
            ])),

            #Caused minor decrease but could perform well on unseen data
            ('pattern_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('num_patterns', PatternTransformer()),
                ('vect', DictVectorizer())
            ])),

            ('frequency_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', TfidfVectorizer(min_df=50))
            ])),

            ('bigrams_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('bigrams', CountVectorizer(ngram_range=(1,3)))
            ])),
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        # pprint(len(data['data']))
        # print(data['data'][0])
        # print(Counter([i['label'] for i in data['data']]))
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))


    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    # pprint(feat_train)
    # print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.001, max_iter=15000, shuffle=True, verbose=2)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)

    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier


    ######    WARNING    ######
    ## Code for determining best alpha [Best alpha = 0.001]
    ## Note: The below code takes ~14hrs if we keeps 15000 iterations for all alphas
    ## Result is mentioned in the solution pdf
    start = time.time()
    max_accuracy = 0
    max_alpha = None
    for alpha_ in [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3]:
        print("Alpha = {}".format(alpha_))
        # Train classifier
        lr = SGDClassifier(loss='log', penalty='l2', alpha=alpha_, max_iter=15000, shuffle=True, verbose=0)

        # K-fold cross validation
        skf = StratifiedKFold(n_splits=5)
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        count = 0
        avg_accuracy_train = 0.0
        avg_accuracy_test = 0.0
        for train, test in skf.split(dataset_x, dataset_y):
            count += 1
            print("Iteration : {}".format(count))
            feat_train = feat.train_feature({
                'text': [t for t in dataset_x[train]]
            })
            # Here we collect the test features
            feat_test = feat.test_feature({
                'text': [t for t in dataset_x[test]]
            })

            lr.fit(feat_train, dataset_y[train])
            y_pred = lr.predict(feat_train)
            accuracy = accuracy_score(y_pred, dataset_y[train])
            avg_accuracy_train += accuracy
            print("Accuracy on training set =", accuracy)

            y_pred = lr.predict(feat_test)
            accuracy = accuracy_score(y_pred, dataset_y[test])
            avg_accuracy_test += accuracy
            print("Accuracy on test set =", accuracy)

        print("Avg. accuracy on training set =", avg_accuracy_train/5)
        print("Avg. accuracy on test set =", avg_accuracy_test/5)
        print("")

        if avg_accuracy_test/5 > max_accuracy:
            max_accuracy = avg_accuracy_test/5
            max_alpha = alpha_

    print("\n\nMaximum : ", end="")
    print(max_accuracy, max_alpha)
    print("Time taken  = ", time.time() - start)