import argparse
import csv
import nltk
from collections import Counter, Mapping, Sequence, defaultdict
from itertools import groupby
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool
from dateutil import parser as dateparser
import os
import sys
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import matplotlib.pyplot as plt
# import plotly.plotly as py

parser = argparse.ArgumentParser("scale the feature vector from input file")
parser.add_argument("trainData") #feature_scaled.csv
parser.add_argument("result") #result.csv

args = parser.parse_args()
CLASSES = ['not a real question', 'not constructive', 'off topic', 'open', 'too localized']

print ('reading input feature vector...')
features= pd.read_csv(args.trainData, nrows=101)
print ('read input feature vector...')

print('converting dict to numpy array..')
# features dict to numpy array
features_array= np.array(features)

# train= features_array[:, :-1]
# output= testFeatures_array[:, -1]   #last column is the output

# test= testFeatures_array[:, :-1]

train, test, trainOutput, testOutput= cross_validation.train_test_split(features_array[:, :-1],
        features_array[:, -1], test_size= 0.2, random_state=1)

# print train

print 'starting to train on linear svm....'
#training on linear svm
linear_svc = svm.LinearSVC()
clf= linear_svc.fit(train, trainOutput)

print 'linear svm trained...'

output= linear_svc.predict(test)
# writing new csv in feature_scaled.csv using pandas
print "predicting output done.. mapping it to values now.."

def getStatus(x): return CLASSES[int(x)]
outputFinal= list(map(getStatus, output))

print "saving to result file now."
outputFinal= np.array(outputFinal)

np.savetxt(args.result, outputFinal, fmt='%s')
print clf.score(test, testOutput)

features_weight= np.zeros((1,clf.coef_.shape[1]))

for x in range(0, clf.coef_.shape[1]):
    features_weight[0,x]= np.mean(clf.coef_[:,x])

features_weight= features_weight

min= np.min(features_weight)
max= np.max(features_weight)

# for i in range(0, features_weight.shape[1]-1):
#     features_weight[0, i]=

for i in range(0, features_weight.shape[1]):
    features_weight[0, i]= (features_weight[0, i] - min)*9/(max-min+1)

# features_weight= (features_weight-min)*9/(max-min)+1
# print len(features_weight.shape[1])

# plot bar graph
keys = ('num_sent', 'num_question', 'num_exclam', 'num_period', 'num_istart', 'num_initcap', 'num_digit',
                  'num_url', 'num_nonword', 'num_finalthanks', 'num_codeblock', 'num_textblock', 'num_lines',
                  'num_tags', 'len_title', 'len_text', 'len_code', 'len_firsttext', 'len_firstcode', 'len_lasttext',
                  'len_lastcode', 'ratio_tc', 'ratio_ftc', 'ratio_ftext', 'ratio_fcode', 'ratio_qsent', 'ratio_esent',
                  'ratio_psent', 'mean_code', 'mean_text', 'mean_sent', 'user_age', 'user_reputation',
                  'user_good_posts', 'user_userid')
width=0.35
y_pos= np.arange(len(keys))
# print y_pos
plt.barh(y_pos, np.transpose(features_weight), width, color="g")
plt.yticks(y_pos, keys)
plt.xlabel('anshul')

plt.show()

