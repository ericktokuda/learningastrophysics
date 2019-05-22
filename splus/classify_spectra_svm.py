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
                  cache_size=5000,
                  class_weight=None,
                  verbose=False,
                  max_iter=-1,
                  decision_function_shape='ovr',
                  random_state=0)
    clf.fit(x_train, y_train) 
    pred_val = clf.predict(x_val)
    prec = confusion_matrix(y_val, pred_val)
    confusion = prec/prec.sum(axis=1)
    return clf, confusion

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paramspath', required=True, help='Path to the params file')
    parser.add_argument('--outpath', required=False, default='/tmp/results.log', help='Output log ')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG, filename=args.outpath)

    # Parse input
    catalogfull = pd.read_csv('data/early_dr.csv')
    params = pd.read_csv(args.paramspath)

    # Filter by r band
    #catalogfull = catalogfull[(catalogfull.r >= 16) & (catalogfull.r < 19)]

    # Bands names
    narrownames = 'f378,f395,f410,f430,f515,f660,f861'.split(',')
    broadnames = 'u,g,r,i,z'.split(',')

    mybands = narrownames + broadnames

    # Assuming they are already split
    #x_train = trainfull[mybands]
    #x_val = valfull[mybands]
    x_catalog = catalogfull[mybands]
    
    #x_train = preprocessing.scale(x_train)
    #x_val = preprocessing.scale(x_val)
    #x_test = preprocessing.scale(x_test)

    # Names -> labels
    le = preprocessing.LabelEncoder()
    le.fit(catalogfull['class'])
    classes = le.classes_
    #y_train = le.transform(trainfull['class'])
    #y_val = le.transform(valfull['class'])
    y_catalog = le.transform(catalogfull['class'])

    spectra = pickle.load(open('/home/keiji/temp/spectra.pkl', 'rb'))
    #print(spectra)
    #print(spectra.columns)
    spectraids = spectra.index
    spectraidsset = set(spectraids)
    catalogidsset = set(catalogfull.id)
    input('spectraidsset:{}'.format(spectraidsset))
    input('catalogidsset:{}'.format(catalogidsset))
    input(len(spectraidsset.difference(catalogidsset)))
    input(len(spectraidsset))
    #print((spectraidsset.difference(catalogidsset)))
    #print(catalogfull.columns)
    #print(catalogfull.id.merge(spectraids) == len(spectraids))
    return
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
