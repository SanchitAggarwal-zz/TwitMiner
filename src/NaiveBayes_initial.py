# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:33:41 2013

@author: learner
"""

import nltk
import itertools
import sys
import random
import os
import re
import nltk.classify.svm

class Classifier(object):
    """classify by looking at a site"""
    def __init__(self, training_set):
        self.training_set = training_set
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.stemmer = nltk.PorterStemmer()
        self.minlength = 7
        self.maxlength = 25

    def text_process_entry(self, example):
        site_text = nltk.clean_html(example[0]).lower()
        original_tokens = itertools.chain.from_iterable(nltk.word_tokenize(w) for w in nltk.sent_tokenize(site_text))
        #original_tokens = [nltk.word_tokenize(w) for w in nltk.sent_tokenize(site_text)]
        tokens = original_tokens# + [' '.join(w) for w in nltk.util.ngrams(original_tokens, 2)]
        #tokens = nltk.util.ngrams(original_tokens, 2)
        #tokens = [w for w in original_tokens + nltk.util.ngrams(original_tokens, 2)]
        #print  nltk.util.ngrams(original_tokens, 2)
        tokens = [w for w in tokens if not w in self.stopwords]
        tokens = [w for w in tokens if self.minlength < len(w) < self.maxlength]
        #print  nltk.util.ngrams(tokens, 2)
        #tokens = [self.stemmer.stem(w) for w in tokens]
        return (tokens, example[1])

    def text_process_all(self, exampleset):
        processed_training_set = [self.text_process_entry(i) for i in self.training_set]
        processed_training_set = filter(lambda x: len(x[0]) > 0, processed_training_set) # remove empty crawls
        processed_texts = [i[0] for i in processed_training_set]
        
        all_words = nltk.FreqDist(itertools.chain.from_iterable(processed_texts))
        features_to_test = all_words.keys()[:5000]
        self.features_to_test = features_to_test
        featuresets = [(self.document_features(d), c) for (d,c) in processed_training_set]
        return featuresets

    def document_features(self, document):
        #document_words = set(document)
        features = {}
        for word in self.features_to_test:
            #features['contains(%s)' % word] = (word in document_words)
            features['contains(%s)' % word] = (word in document)
            #features['occurrencies(%s)' % word] = document.count(word) 
            #features['atleast3(%s)' % word] = document.count(word) > 3
        return features

    def build_classifier(self, featuresets):
        random.shuffle(featuresets)
        cut_point = len(featuresets) / 5
        train_set, test_set = featuresets[cut_point:], featuresets[:cut_point]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        #classifier = nltk.classify.svm.SvmClassifier.train(train_set)
        return (classifier, test_set)

    def run(self):
        featuresets = self.text_process_all(self.training_set)
        classifier, test_set = self.build_classifier(featuresets)
        self.classifier = classifier
        self.test_classifier(classifier, test_set)

    def classify(self, text):
        return self.classifier.classify(self.document_features(text))

    def test_classifier(self, classifier, test_set):
        print nltk.classify.accuracy(classifier, test_set)
        classifier.show_most_informative_features(45)
        

def remove_html_tags(s):
    return re.sub(r'\s*<(.|\s)*?>', r'', s) # *? means not greedy

def convert_html_entities(s):
    s = re.sub(r'&quot;', r'"', s)
    s = re.sub(r'&amp;', r'&', s)
    s = re.sub(r'&nbsp;', r' ', s)
    s = re.sub(r'&lt;', r'<', s)
    s = re.sub(r'&gt;', r'>', s)
    return s

def remove_urls(s):
    pattern = r'(?i)(https?\://)?[a-z0-9][\w\-]*(\.[a-z]{2,3})+(\/[\w\:\-#%&\?\/]+)?'
    return re.sub(pattern, r' ', s)

def trim_between(s):
    '''Return string s with all consecutive whitespace removed, as removed whitespace
at beginning and end of s'''

    s = re.sub(r'(\t)+', r'\1', s.strip())
    s = re.sub(r'( )+', r'\1', s)
    s = re.sub(r'(\n)+', r'\1', s)
    return s
    
def readTrainingData(filename):     #containing the class label followed by the features, 
    os.path.relpath(filename,os.curdir)         #separated by blank spaces.
    Data=[]
    count=0
    try:          
        with open(filename, 'r') as f:
            for line in f:
                if(count==5500): break
                line=line.split(' ', 2)
                tweet=line.pop()
                tweet = convert_html_entities(tweet)
                tweet = remove_urls(tweet)
                tweet = remove_html_tags(tweet)
                tweet = trim_between(tweet)
                label=line.pop()
                data_item=(tweet,label)
                Data.append(data_item)
                count=count+1
        return Data
    except:
        print "exception in read data",sys.exc_info()
        return False

def readValidationData(filename):     #containing the class label followed by the features, 
    os.path.relpath(filename,os.curdir)         #separated by blank spaces.
    Data=[]
    count=0
    try:          
        with open(filename, 'r') as f:
            for line in f:
                #if(count==5): break
                line=line.split(' ', 1)
                tweet=line.pop()
                tweet = convert_html_entities(tweet)
                tweet = remove_urls(tweet)
                tweet = remove_html_tags(tweet)
                tweet = trim_between(tweet)
                tweet_id=line.pop()
                Data.append((tweet,tweet_id))
                count=count+1
        return Data
    except:
        print "exception in reading validation data",sys.exc_info()
        return False

def writeData(twitter_output):
    filename="Austere_output.txt"
    try:
        os.path.relpath(filename,os.curdir)
        File_Write = open(filename, "w+")           
        for item in twitter_output:
                File_Write.write(item)
        File_Write.close()
        return True
    except:
        print "exception in write data"
        return False

classes = ('Sports', 'Politics')
training_set=readTrainingData('training.txt')
validation_set=readValidationData('validation.txt')
print len(training_set),len(validation_set)
#print training_set[0][1]



if __name__ == '__main__':
    classifier = Classifier(training_set)
    classifier.run()
    twitter_output=[]
    for test_text in validation_set:
        item=str(test_text[1])+' '+classifier.classify(test_text[0])+"\n"
        twitter_output.append(item)
    writeData(twitter_output)
    print "finish"
