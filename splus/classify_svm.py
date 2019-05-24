#!/usr/bin/env python3
"""Classify using a pool of svm classifiers with different params
for F in data/params_*; do X=$(basename $F) && X=${X/csv/log} &&  echo python classify_svm.py data/spectra.pkl --trainids data/train_spectra.txt --valids data/val_spectra.txt  --testids data/test_spectra.txt --cols data/cols.txt --paramspath $F --outpath results/spectra/$X & done
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix
import time
import logging
from logging import debug
import pickle


def svm_trainval(x_train, y_train, x_val, y_val, myc, myker, mydeg, mygamma):
    """Train and evalute svm

    Args:
    x_train, x_val(np.ndarray): row represent sample and column, the feature
    y_train, y_val(np.ndarray): row represent sample and column, the gt label
    myc(float): SVM C
    myker(str): SVM kernel (see sklearn svm)
    mydeg(int): degree for polynomial svm
    mygamma(float): SVM gamma

    """

    clf = svm.SVC(C=myc,
                  kernel=myker,
                  degree=mydeg,
                  gamma=mygamma,
                  coef0=0.0,
                  shrinking=True,
                  probability=False,
                  tol=0.001,
                  cache_size=48000,
                  class_weight=None,
                  verbose=False,
                  max_iter=-1,
                  decision_function_shape='ovr',
                  random_state=0)
    clf.fit(x_train, y_train) 
    pred_val = clf.predict(x_val)
    prec = confusion_matrix(y_val, pred_val)
    cm = prec.astype('float') / prec.sum(axis=1)[:, np.newaxis]
    return clf, cm

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('featurespath', help='Features vector in .pkl format')
    parser.add_argument('--paramspath', required=False, default='data/params.csv', help='Path to the params file')
    parser.add_argument('--trainids', required=False, default='data/train.txt', help='Train list path')
    parser.add_argument('--valids', required=False, default='data/val.txt', help='Validation list path')
    parser.add_argument('--testids', required=False, default='data/test.txt', help='Test list path')
    parser.add_argument('--cols', required=False, default='data/cols.txt', help='Columns list path')
    parser.add_argument('--outpath', required=False, default='/tmp/results.log', help='Output log ')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG, filename=args.outpath)

    cols = np.loadtxt(args.cols, dtype=str)
    trainids = np.loadtxt(args.trainids, dtype=str)
    valids = np.loadtxt(args.valids, dtype=str)
    testids = np.loadtxt(args.testids, dtype=str)
    params = pd.read_csv(args.paramspath)
    loaded = pickle.load(open(args.featurespath, 'rb'))

    features_train = loaded[loaded.id.isin(trainids)]
    features_val = loaded[loaded.id.isin(valids)]
    features_test = loaded[loaded.id.isin(testids)]

    x_train = features_train[cols]
    x_val = features_val[cols]
    x_test = features_test[cols]

    # Names -> labels
    le = preprocessing.LabelEncoder()
    le.fit(features_train['class'])
    classes = le.classes_
    debug(classes)

    y_train = le.transform(features_train['class'])
    y_val = le.transform(features_val['class'])
    y_test = le.transform(features_test['class'])

    # Grid search with different kernels
    params_best = []
    prec_best = -1
    for idx, p in params.iterrows():
        myker, mydeg, mygamma, myc = p
        t0 = time.time()
        clf, confusion = svm_trainval(x_train, y_train, x_val, y_val, myc, myker, mydeg, mygamma)
        prec_avg = np.mean(np.diagonal(confusion))
        finalstr = '{},{},{}'.format(','.join(map(str, p)), prec_avg, time.time() - t0)
        debug(finalstr)

        if prec_avg > prec_best:
            prec_best = prec_avg
            params_best = myker, mydeg, mygamma, myc
            
    # Run with test set the best params previously obtained
    myker, mydeg, mygamma, myc = params_best
    clf, confusion = svm_trainval(x_train, y_train, x_test, y_test, myc, myker, mydeg, mygamma)
    debug(','.join(map(str, params_best)))
    debug(confusion)

if __name__ == "__main__":
    main()
