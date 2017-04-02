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

CLASSES = ['not a real question', 'not constructive', 'off topic', 'open', 'too localized']
status = dict((k, str(i)) for i, k in enumerate(CLASSES))

RE_NONALNUM = re.compile(r'\W+')
RE_NONANS = re.compile(r'[^\w\s]+')
RE_DIGIT = re.compile(r'\d+')
RE_URL = re.compile(r'https?://')
RE_NONWORD = re.compile(r'[A-Z\d]+')

#global array of feature heads
fieldnames = ['num_sent', 'num_question', 'num_exclam', 'num_period', 'num_istart', 'num_initcap', 'num_digit',
                  'num_url', 'num_nonword', 'num_finalthanks', 'num_codeblock', 'num_textblock', 'num_lines',
                  'num_tags', 'len_title', 'len_text', 'len_code', 'len_firsttext', 'len_firstcode', 'len_lasttext',
                  'len_lastcode', 'ratio_tc', 'ratio_ftc', 'ratio_ftext', 'ratio_fcode', 'ratio_qsent', 'ratio_esent',
                  'ratio_psent', 'mean_code', 'mean_text', 'mean_sent', 'user_age', 'user_reputation',
                  'user_good_posts', 'user_userid', 'post_status']  # field names/ headers in args.output

def norm(string):
    return RE_NONANS.sub('', string).lower()


def norm_tag(string):
    return RE_NONALNUM.sub('', string).lower()


def ratio(x, y):
    if y != 0:
        return x / float(y)
    else:
        return 0


# function to convert numpy array to dictionary for feature extraction
def conv_to_dict(row):

    keys = ['PostId', 'PostCreationDate', 'OwnerUserId', 'OwnerCreationDate', 'ReputationAtPostCreation',
            'OwnerUndeletedAnswerCountAtPostTime',
            'Title', 'BodyMarkdown', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'PostClosedDate', 'OpenStatus']
    count = 0
    ret_dict = {}
    for element in np.nditer(row, flags=['refs_ok']):
        element = str(element)

        if element != 'nan':
            ret_dict[keys[count]] = element
        else:
            ret_dict[keys[count]] = ''

        count += 1
    return ret_dict


def get_feature_arr(row):
    post_id = row['PostId']
    try:
        post_status = status[row['OpenStatus']]
    except KeyError:
        # no OpenStatus, must be a test file
        post_status = '0'

    title = row['Title']
    body = row['BodyMarkdown']
    tags = [norm_tag(row["Tag%d" % i]) for i in range(1, 6) if row["Tag%d" % i]]

    lines = body.splitlines()
    code = []
    text = []
    sents = []

    # Divide post into code and text blocks
    for is_code, group in groupby(lines, lambda l: l.startswith('    ')):
        (code if is_code else text).append('\n'.join(group))

    # Let's build some features!
    feature = {}

    feature['num_sent'] = 0
    feature['num_question'] = 0
    feature['num_exclam'] = 0
    feature['num_period'] = 0
    feature['num_initcap'] = 0
    feature['num_istart'] = 0
    feature['num_url'] = 0
    feature['num_digit'] = 0
    feature['num_nonword'] = 0
    body_words = set()

    firstsentwords = None
    lastsentwords = None

    for t in text:
        for sent in nltk.sent_tokenize(t):

            feature['num_sent'] += 1
            ss = sent.strip()

            if ss:
                if ss.endswith('?'):
                    feature['num_question'] += 1
                if ss.endswith('!'):
                    feature['num_exclam'] += 1
                if ss.endswith('.'):
                    feature['num_period'] += 1
                if ss.startswith('I '):
                    feature['num_istart'] += 1
                if ss[0].isupper():
                    feature['num_initcap'] += 1

            words = nltk.word_tokenize(norm(sent))

            # We track the set of words of the first and last sentences
            lastsentwords = set(words)
            if firstsentwords is None:
                firstsentwords = lastsentwords
            body_words |= lastsentwords
            sents.append(ss)

        feature['num_digit'] += len(RE_DIGIT.findall(t))
        feature['num_url'] += len(RE_URL.findall(t))
        feature['num_nonword'] += len(RE_NONWORD.findall(t))

    feature['num_finalthanks'] = 1 if text and 'thank' in text[-1].lower() else 0

    body_words = list(body_words)
    firstsentwords = list(firstsentwords) if firstsentwords else []
    lastsentwords = list(lastsentwords) if lastsentwords else []

    title_words = nltk.word_tokenize(norm(title))
    title_words = list(set(title_words))

    post_t = dateparser.parse(row['PostCreationDate'])
    user_t = dateparser.parse(row['OwnerCreationDate'])

    # Some information about the user
    feature['user_age'] = (post_t - user_t).total_seconds()
    feature['user_reputation'] = int(row['ReputationAtPostCreation'])
    feature['user_good_posts'] = int(row['OwnerUndeletedAnswerCountAtPostTime'])
    feature['user_userid'] = row['OwnerUserId']

    feature['num_codeblock'] = len(code)
    feature['num_textblock'] = len(text)
    feature['num_lines'] = len(lines)
    feature['num_tags'] = len(tags)
    feature['len_title'] = len(title)
    feature['len_text'] = sum(len(t) for t in text)
    feature['len_code'] = sum(len(c) for c in code)
    feature['len_firsttext'] = len(text[0]) if text else 0
    feature['len_firstcode'] = len(code[0]) if code else 0
    feature['len_lasttext'] = len(text[-1]) if text else 0
    feature['len_lastcode'] = len(code[-1]) if code else 0
    feature['ratio_tc'] = ratio(feature['len_text'], feature['len_code'])
    feature['ratio_ftc'] = ratio(feature['len_firsttext'], feature['len_firstcode'])
    feature['ratio_ftext'] = ratio(feature['len_firsttext'], feature['len_text'])
    feature['ratio_fcode'] = ratio(feature['len_firstcode'], feature['len_code'])
    feature['ratio_qsent'] = ratio(feature['num_question'], feature['num_sent'])
    feature['ratio_esent'] = ratio(feature['num_exclam'], feature['num_sent'])
    feature['ratio_psent'] = ratio(feature['num_period'], feature['num_sent'])
    feature['mean_code'] = np.mean([len(c) for c in code]) if code else 0
    feature['mean_text'] = np.mean([len(t) for t in text]) if text else 0
    feature['mean_sent'] = np.mean([len(s) for s in sents]) if sents else 0
    feature['post_status'] = post_status

    for key in feature:
        feature[key] = float(feature[key])


    # into features array
    return feature


# main code body
if __name__ == "__main__":
    parser = argparse.ArgumentParser("build the feature vector from train.csv")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    # preparations for writing into file
    write_file = open(args.output, 'wb')
    csvwriter = csv.DictWriter(write_file, delimiter=',', fieldnames=fieldnames)
    csvwriter.writerow(dict((fn, fn) for fn in fieldnames))

    # reading in input file
    for chunk in pd.read_csv(args.input, chunksize=1000):
        arr = np.array(chunk)
        for entry in arr:
            try:
                print "\n"
                feature_arr = get_feature_arr(conv_to_dict(entry))
                csvwriter.writerow(feature_arr)
                print ("In csv")
                print feature_arr
                print "\n"
            except UnicodeDecodeError:
                print("Sorry")
        arr = []

    # arr1= pd.read_csv(args.input, nrows=1001)
    # arr2= np.array(arr1)
    # for entry in arr2:
    #     try:
    #         print "\n"
    #         feature_arr = get_feature_arr(conv_to_dict(entry))

    #         print entry
    #         csvwriter.writerow(feature_arr)
    #         # print ("In csv")
    #         # print feature_arr
    #         print "\n"
    #     except UnicodeDecodeError:
    #          print("Sorry")
