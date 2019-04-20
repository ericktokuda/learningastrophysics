#!/usr/bin/env python3
"""Classify catalog
"""

import argparse
import os
from os.path import join as ojoin
import time
import logging
from logging import debug
import pickle
import itertools
import numpy as np
import datetime

import matplotlib
from matplotlib import pyplot as plt

import pandas as pd

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn import preprocessing


def read_catalogs_dir(catalogdir, cols, _ids=None):
    """Loop through each file in @catalogdir and keep just the ids in @_ids

    Args:
    catalogdir(str): path to the catalogs
    cols(list): list of column names
    _ids(list): list of ids to keep. Default value considers all ids.

    Returns:
    pd.Dataframe: rows of the ids filtered by @_ids
    """

    ids = _ids
    features = pd.DataFrame()

    for f in sorted(os.listdir(catalogdir)):
        catalogpath = ojoin(catalogdir, f)
        if os.path.isdir(catalogpath): continue

        debug(catalogpath)
        df = pd.read_csv(catalogpath, sep='\s+', header=None, names=cols,
                         skiprows=191, engine='python')
        
        if not _ids:
            features = features.append(df, ignore_index=True)
        else:
            newrows = df[df.ID.isin(ids)]
            ids = ids.difference(set(newrows.ID))
            features = features.append(newrows, ignore_index=True)
    return features.sort_values(by='ID')

def read_headers_and_identifiers(headerpath):
    """Read column names, broad band identifiers and narrow band identifiers.
    These are catalog-specific data.

    Args:
    headerpath(str): path to the file

    Returns:
    3-uple of lists: column names, broadbands suffixes and narrowband suffixes
    """
    with open(headerpath) as fh:
        fh.readline()
        cols = fh.readline().strip().split(',')
        broad_suff = fh.readline().strip().split(',')
        narrow_suff = fh.readline().strip().split(',')
    return cols, broad_suff, narrow_suff

def apply_regression(x_train, y_train, x_test, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    #logging.debug('Coefficients: \n', regr.coef_)
    debug("Mean squared error: {:2f}\tVariance score: {:2f}"
          .format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

def get_catalogs(catalogspath, catalogspklpath, cols, ids_keep):
    """Read csv or load pickle file from catalogs filtered by @ids_keep

    Args:
    catalogspath(str): path to the catalogs dir
    catalogspklpath(str): path to the pickle file,
    cols(list): list of column names,
    ids_keep(list): list of IDS to keep

    Returns:
    pandas.Dataframe: catalogs matrix
    """

    if not os.path.exists(catalogspklpath):
        features = read_catalogs_dir(catalogspath, cols, ids_keep)
        pickle.dump(features, open(catalogspklpath, 'wb'))
    else:
        features = pickle.load(open(catalogspklpath, 'rb'))
    return features

def cv_svm(x, y, cv, params):
    """Cross-validation evaluation using SVM

    Args:
    x(2d tensor): (i, j) element represent j-th atribute of the i-th sample
    y(list): ground-truth
    cv(ShuffleSplit): sklearn object to handle cross validation
    params(pandas.DataFrame: SVM parameters

    Returns:
    ndarray: (i, j) element represent i-th score from the i-th run
    """


    k = cv.get_n_splits()
    allscores = np.ndarray((0, k))
    times = []

    nrows = params.shape[0]
    for idx, (myker, mydeg, mygamma, myc) in params.iterrows():
        debug('##########################################################')
        debug('Cross-validation {}/{}...'.format(idx, nrows))
        
        t0 = time.time()
        
        myc, myker, mydeg, mygamma
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
                      max_iter=2000,
                      decision_function_shape='ovr',
                      random_state=0)
        debug(clf)
        scores = cross_val_score(clf, x, y, cv=cv)
       
        debug(scores)
        allscores = np.concatenate((allscores, scores.reshape(1, k)), axis=0)
        elapsed = time.time() - t0
        debug('Elapsed time: {:.2f}s'.format(time.time() - t0))
        times.append(elapsed)
    return allscores

def dump_results(broadscores, narrowscores, unionscores, params, resultsdir):
    """Output results to csv

    Args:
    broadscores, narrowscores, unionscores (ndarray): scores
    params(pandas.DataFrame): parameters
    resultsdir(str): path to the results directory

    Returns:
    None
    """

    pickle.dump(broadscores, open(ojoin(resultsdir, 'broadscores.pkl'), 'wb'))
    pickle.dump(narrowscores, open(ojoin(resultsdir, 'narrowscores.pkl'), 'wb'))
    pickle.dump(unionscores, open(ojoin(resultsdir, 'unionscores.pkl'), 'wb'))
    n = len(params)
    bm = broadscores.mean(axis=1).reshape(n, 1)
    bs = broadscores.std(axis=1).reshape(n, 1)
    nm = narrowscores.mean(axis=1).reshape(n, 1)
    ns = narrowscores.std(axis=1).reshape(n, 1)
    um = unionscores.mean(axis=1).reshape(n, 1)
    us = unionscores.std(axis=1).reshape(n, 1)
    summary = np.concatenate([np.array(params), bm, bs, nm, ns, um, us], axis=1)
    h = 'C,deg,gamma,ker,broadmean,broadstd,narrowmean,narrowstd,unionmean,unionstd'
    np.savetxt(ojoin(resultsdir, 'summary.csv'), summary, fmt='%s', delimiter=',',
               header=h, comments='')
    debug(summary)

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogspath', required=True, help='Path to the catalogs dir')
    parser.add_argument('--gtpath', required=True, help='Path to the ground-truth csv')
    parser.add_argument('--paramspath', required=False, help='Path to the parameters file',
                        default='data/params.csv')
    args = parser.parse_args()

    headerpath = 'data/header.txt'
    resultsdir = ojoin('results', datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    os.mkdir(resultsdir)
    params = pd.read_csv(args.paramspath)

    logging.basicConfig(level=logging.DEBUG)

    with open(ojoin(resultsdir, 'README.md'), 'w') as fh:
        fh.write(args.gtpath + '\n')
        fh.write(args.catalogspath + '\n')
        fh.write(headerpath + '\n')

    classesfname = os.path.basename(args.gtpath)
    catalogspklpath = ojoin('data/', os.path.splitext(classesfname)[0] + '.pkl')

    cols, broadbands, narrowbands = read_headers_and_identifiers(headerpath)
    classes = pd.read_csv(args.gtpath).drop_duplicates(subset='id')
    features = get_catalogs(args.catalogspath, catalogspklpath, cols, set(classes.id))

    # Prepare mapping (classes->number)
    classes_map = {}
    for acc, a in enumerate(set(classes['class'])): classes_map[a] = acc

    # Prepare Y
    gt = classes[['id', 'class']]
    gt = gt[gt.id.isin(features.ID)]
    gt['class'] = gt['class'].map(classes_map)
    gt = gt.sort_values(by='id')
    y = gt['class'].values

    # Prepare X (scale)
    x_broad = preprocessing.scale(features[broadbands])
    x_narrow = preprocessing.scale(features[narrowbands])
    x_union = preprocessing.scale(features[broadbands+narrowbands])

    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

    broadscores = cv_svm(x_broad, y, cv, params)
    narrowscores = cv_svm(x_narrow, y, cv, params)
    unionscores = cv_svm(x_union, y, cv, params)

    dump_results(broadscores, narrowscores, unionscores, params, resultsdir)

##########################################################
if __name__ == "__main__":
    main()
