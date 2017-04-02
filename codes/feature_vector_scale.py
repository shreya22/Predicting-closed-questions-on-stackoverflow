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
from sklearn import preprocessing


# main code body
#wil be run only if called as the main program and not from any other module
if __name__ == "__main__":
    parser = argparse.ArgumentParser("scale the feature vector from output.csv")
    parser.add_argument("input") #output.csv
    parser.add_argument("output") #feature_scaled.csv
    args = parser.parse_args()

    print ('read input feature vector')
    features= pd.read_csv(args.input)

    # features dict to numpy array
    features_array= np.array(features)

    # removing last row, as it was giving nan -_-
    features_array= features_array[:-1, :]

    # num of columns, excluding the last column of open status
    num_columns= features_array.shape[1]-1

    #initialising mu and sigma
    mu= np.zeros((num_columns, 1))
    sigma= np.zeros((num_columns, 1))

    # calculate mean in mu and standard deviation in sigma
    for i in range (0, num_columns):
        mu[i][0]= np.mean(features_array[:, i])
        sigma[i][0]= np.std(features_array[:, i])
        sigma[i][0]= 1 if sigma[i][0]==0 else sigma[i][0]

    # X= (X-mu)/sigma
    for i in range(0, num_columns):
        features_array[:-1, i]= (features_array[:-1, i] - mu[i])/sigma[i]
        print i


    # writing new csv in feature_scaled.csv using pandas
    np.savetxt(args.output, features_array, delimiter=",")
