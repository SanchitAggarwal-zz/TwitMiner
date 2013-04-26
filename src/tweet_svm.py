# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:55:59 2013

@author: learner
"""
import re,os,sys
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk.classify

Sports_Tweet=[]
Politics_Tweet=[]
Stopwords = stopwords.words("english")
minlength = 4
maxlength = 25


def evaluate_classifier(featx):
    
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    sportswords =[]
    for tweet in Sports_Tweet:    
        sportswords=sportswords+featx(tweet[0]) 
  
    politicswords =[]
    for tweet in Politics_Tweet:
        politicswords=politicswords+featx(tweet[0]) 
    
    for word in sportswords:
        word_fd.inc(word)
        label_word_fd['Sports'].inc(word)
 
    for word in politicswords:
        word_fd.inc(word)
        label_word_fd['Politics'].inc(word)
 
    sports_word_count = label_word_fd['Sports'].N()
    politics_word_count = label_word_fd['Politics'].N()
    total_word_count = sports_word_count + politics_word_count

    sportsfeats=[]
    politicsfeats=[]
    '''for word, freq in word_fd.iteritems():
        sports_score = BigramAssocMeasures.chi_sq(label_word_fd['Sports'][word],
                                        (freq, sports_word_count), total_word_count)
        sportsfeats=sportsfeats+[ ( {word:label_word_fd['Sports'][word]}, str('Sports') ) ]
        
        politics_score = BigramAssocMeasures.chi_sq(label_word_fd['Politics'][word],
                                            (freq, politics_word_count), total_word_count)
        politicsfeats=politicsfeats+[ ( {word:label_word_fd['Politics'][word]}, str('Politics') ) ]'''
    
    for word, freq in word_fd.iteritems():
        sports_score=label_word_fd['Sports'][word]
        politics_score=label_word_fd['Politics'][word]
        #if sports_score > politics_score :
        sportsfeats=sportsfeats+[ ( {word:sports_score}, str('Sports') ) ]
        #else:        
        politicsfeats=politicsfeats+[ ( {word:politics_score}, str('Politics') ) ]
    
    '''sportsfeats = [(featx(tweet[0]), tweet[1]) for tweet in Sports_Tweet]
    politicsfeats = [(featx(tweet[0]), tweet[1]) for tweet in Politics_Tweet]'''
    
    sportscutoff = len(sportsfeats)*3/4
    politicscutoff = len(politicsfeats)*3/4
 
    trainfeats = sportsfeats[:sportscutoff] + politicsfeats[:politicscutoff]
    testfeats = sportsfeats[sportscutoff:] + politicsfeats[politicscutoff:]
 

    print('--- nltk.classify.svm demo ---')
    print('Number of training examples:', len(trainfeats))
    classifier = nltk.classify.svm.SvmClassifier.train(trainfeats)
    print('Total SVM dimensions:', len(classifier._svmfeatureindex))
    print('Label mapping:', classifier._labelmapping)
    print('--- Processing an example instance ---')
    print('NLTK-format features:\n    ' + str(testfeats[0]))
    print('SVMlight-format features:\n    ' + str(nltk.classify.svm.map_instance_to_svm(testfeats[0], classifier._labelmapping, classifier._svmfeatureindex)))
    distr = classifier.prob_classify(testfeats[0][0])
    print('Instance classification and confidence:', distr.max(), distr.prob(distr.max()))
    print('--- Measuring classifier performance ---')
#    print('Overall accuracy:', nltk.classify.util.accuracy(classifier, testfeats)) 

 
    #classifier = NaiveBayesClassifier.train(trainfeats)
    '''refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    print 'pos precision:', nltk.metrics.precision(refsets['Sports'], testsets['Sports'])
    print 'pos recall:', nltk.metrics.recall(refsets['Politics'], testsets['Politics'])
    print 'neg precision:', nltk.metrics.precision(refsets['Sports'], testsets['Sports'])
    print 'neg recall:', nltk.metrics.recall(refsets['Politics'], testsets['Politics'])
    classifier.show_most_informative_features()'''
    return classifier,word_fd,label_word_fd, sports_word_count,politics_word_count,total_word_count
 
def word_feats(words):
     words = nltk.clean_html(words).lower()
     original_tokens = itertools.chain.from_iterable(nltk.word_tokenize(w) for w in nltk.sent_tokenize(words))
     #original_tokens = [w for w in original_tokens if not w in Stopwords]
     original_tokens = [w for w in original_tokens if minlength < len(w) < maxlength]
     #return {'last_letter': word[-1], 'penultimate_letter': word[-2]}     
     #print { original_tokens[i]:i for i in range (0,len(original_tokens))}
     return original_tokens
def clean_tweet(s):
    s= re.sub(r'\s*<(.|\s)*?>', r'', s) # *? means not greedy
    s = re.sub(r'&quot;', r'"', s)
    s = re.sub(r'&amp;', r'&', s)
    s = re.sub(r'&nbsp;', r' ', s)
    s = re.sub(r'&lt;', r'<', s)
    s = re.sub(r'&gt;', r'>', s)
    pattern = r'(?i)(http?\://)?[a-z0-9][\w\-]*(\.[a-z]{2,3})+(\/[\w\:\-#%&\?\/]+)?'
    s= re.sub(pattern, r' ', s)
    s = re.sub(r'(\t)+', r'\1', s.strip())
    s = re.sub(r'( )+', r'\1', s)
    s = re.sub(r'(\n)+', r'\1', s)
    s = re.sub(r'(#)+', r'', s)
    s = re.sub(r'(@)+', r'', s)
    s = re.sub(r'(\\)', r' ', s)
    s = re.sub(r'([0-9]+)', r'', s)
    return s

def readTrainingData(filename):     #containing the class label followed by the features, 
    os.path.relpath(filename,os.curdir)         #separated by blank spaces.
    count=0
    try:          
        with open(filename, 'r') as f:
            for line in f:
                if(count==6500): break
                line=line.split(' ', 2)
                tweet=clean_tweet(line.pop())
                label=line.pop()
                data_item=(tweet,label)
                if label=='Sports':Sports_Tweet.append(data_item)
                else : Politics_Tweet.append(data_item)
                count=count+1
        return True
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
                tweet=clean_tweet(line.pop())
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


readTrainingData('training.txt')
classifier,word_freq,label_freq, sports_word_count,politics_word_count,total_word_count=evaluate_classifier(word_feats)
validation_set=readValidationData('validation.txt')
#validation_set=readValidationData('test.txt')
twitter_output=[]

for test_text in validation_set:
    features=word_feats(test_text[0])
    features_fd = FreqDist(features)
    feats=dict()
    for w,f in features_fd.iteritems():
        total_score=0
        for word, freq in word_freq.iteritems():
            if w==word:
                '''s_score = BigramAssocMeasures.chi_sq(label_freq['Sports'][word],
                                        (freq, sports_word_count), total_word_count)
                p_score = BigramAssocMeasures.chi_sq(label_freq['Politics'][word],
                                            (freq, politics_word_count), total_word_count)
                total_score=s_score+p_score'''
                break
        feats[w] = freq

    distr = classifier.prob_classify(feats)
    p_labels =  classifier.classify(feats)
    print feats,distr.max(),p_labels
    #print distr.samples()
    item=str(test_text[1])+' '+ p_labels+"\n"
    twitter_output.append(item)
writeData(twitter_output)
print "finish"
