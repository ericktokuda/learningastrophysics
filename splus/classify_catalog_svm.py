#!/usr/bin/env python3
"""Classify using a pool of svm classifiers with different params
 for F in data/params*; do X=$(basename $F) && X=${X/csv/log} && echo python classify_catalog_svm.py  --paramspath $F  --outpath /tmp/$X && nohup python classify_catalog_svm.py  --paramspath $F  --outpat /tmp/$X & done
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
                  cache_size=5000,
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
    parser.add_argument('--paramspath', required=True, help='Path to the params file')
    parser.add_argument('--outpath', required=False, default='/tmp/results.log', help='Output log ')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG, filename=args.outpath)

    # Parse input
    trainfull = pd.read_csv('/home/anamartinazzo/raw-data/dr1_full_train.csv')
    valfull = pd.read_csv('/home/anamartinazzo/raw-data/dr1_full_val.csv')
    testfull = pd.read_csv('/home/anamartinazzo/raw-data/dr1_full_test.csv')
    #trainfull = pd.read_csv('data/dr1_full_train_sample.csv')
    #valfull = pd.read_csv('data/dr1_full_val_sample.csv')
    #testfull = pd.read_csv('data/dr1_full_test_sample.csv')
    params = pd.read_csv(args.paramspath)

    # Filter by r band
    #trainfull = trainfull[(trainfull.r >= 16) & (trainfull.r < 19)]
    #valfull = valfull[(valfull.r >= 16) & (valfull.r < 19)]
    #testfull = testfull[(testfull.r >= 16) & (testfull.r < 19)]

    # Bands names
    narrownames = 'f378,f395,f410,f430,f515,f660,f861'.split(',')
    broadnames = 'u,g,r,i,z'.split(',')

    mybands = narrownames + broadnames

    # Assuming they are already split
    x_train = trainfull[mybands]
    x_val = valfull[mybands]
    x_test = testfull[mybands]
    
    #x_train = preprocessing.scale(x_train)
    #x_val = preprocessing.scale(x_val)
    #x_test = preprocessing.scale(x_test)

    # Names -> labels
    le = preprocessing.LabelEncoder()
    le.fit(trainfull['class'])
    classes = le.classes_
    debug(classes)

    y_train = le.transform(trainfull['class'])
    y_val = le.transform(valfull['class'])
    y_test = le.transform(testfull['class'])

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
