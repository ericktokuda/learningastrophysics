#!/usr/bin/env python3
"""Classify catalog
"""

import argparse
import os
import time
import logging
from logging import debug
import pickle
import itertools
import numpy as np
from copy import copy

import matplotlib
from matplotlib import pyplot as plt

import pandas as pd

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.model_selection import ShuffleSplit, cross_val_score

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
        catalogpath = os.path.join(catalogdir, f)
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
    return features

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
        broad_suff = fh.readline().strip().split(', ')
        narrow_suff = fh.readline().strip().split(', ')
    return cols, broad_suff, narrow_suff

def apply_regression(x_train, y_train, x_test, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    #logging.debug('Coefficients: \n', regr.coef_)
    debug("Mean squared error: {:2f}\tVariance score: {:2f}"
          .format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

def get_kth_fold_traintest_indices(k, total, testratio):
    """Partition the indices (testratio, 1-trainratio)

    Args:
    n(int): number of elements
    testratio(float): ratio of the input data for testing

    Returns:
    (list, list): list of train and of test indices
    """

    allindices = set(range(total))
    test = range(k*sz, (k+1)*sz)
    train = allindices.difference(test)
    return train, test

def apply_svm(x, y, cv):
    """Apply svm to the input data

    Args:
    x (2d tensor): (i, j) element represent j-th atribute of the i-th sample
    y (list): ground-truth

    Returns:
    ndarray: (i, j) element represent i-th score from the i-th run
    list: list of tuples of the values of the parameters
    """

    c = [.1, 1, 10]
    degree = [2, 3]
    gamma = ['auto', 'scale']
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    params = itertools.product(c, degree, gamma, kernel)
    paramslist = list(copy(params))

    k = cv.get_n_splits()
    allscores = np.ndarray((0, k))

    for p in params:
        debug('##########################################################')
        debug('Cross-validation for {}...'.format(id))
        
        t0 = time.time()
        
        clf = svm.SVC(C=p[0],
                      cache_size=2000,
                      class_weight=None,
                      coef0=0.0,
                      decision_function_shape='ovr',
                      degree=p[1],
                      gamma=p[2],
                      kernel=p[3],
                      max_iter=2000,
                      probability=False,
                      random_state=None,
                      shrinking=True,
                      tol=0.001,
                      verbose=False)
        debug(clf)
        scores = cross_val_score(clf, x, y, cv=cv)
       
        debug(scores)
        allscores = np.concatenate((allscores, scores.reshape(1, k)), axis=0)
        debug('Elapsed time: {:.2f}s'.format(time.time() - t0))
    return allscores, paramslist

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    #classespath = '/home/frodo/temp/raw-data/sloan_splus_matches.csv'
    classespath = '/home/frodo/temp/raw-data/sloan_splus_matches_sample500.csv'
    catalogdir = '/home/frodo/temp/raw-data/catalogs/'
    headerpath = 'data/header.txt'
    resultsdir = 'results'

    classesfname = os.path.basename(classespath)
    catalogspklpath = os.path.join('data/', os.path.splitext(classesfname)[0] + '.pkl')
    aperture = 'auto' # ['auto'  'petro'  'aper' ]

    cols, broad_suff, narrow_suff = read_headers_and_identifiers(headerpath)
    classes = pd.read_csv(classespath).drop_duplicates(subset='id')

    broadbands = [ x + '_' + aperture for  x in broad_suff ]
    narrowbands = [ y + '_' + aperture for  y in narrow_suff ]

    ids_keep = set(classes.id)

    if not os.path.exists(catalogspklpath):
        features = read_catalogs_dir(catalogdir, cols, ids_keep)
        pickle.dump(features, open(catalogspklpath, 'wb'))
    else:
        features = pickle.load(open(catalogspklpath, 'rb'))

    features_broad = features[['ID'] + broadbands]
    features_narrow = features[['ID'] + narrowbands]
    features_union = features[['ID'] + broadbands + narrowbands]

    classes_map = {}    # Create a map of numbers->classes
    for acc, a in enumerate(set(classes['class'])): classes_map[a] = acc

    gt = classes[['id', 'class']]
    gt = gt[gt.id.isin(features.ID)]
    gt['class'] = gt['class'].map(classes_map)
    gt = gt.sort_values(by='id')
    y = gt['class'].values

    x_broad = features_broad.sort_values(by='ID')[broadbands]
    x_narrow = features_narrow.sort_values(by='ID')[narrowbands] # Narrowbands
    x_union = features_union.sort_values(by='ID')[broadbands + narrowbands] # All bands

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    broadscores, params = apply_svm(x_broad, y, cv)
    narrowscores, _ = apply_svm(x_narrow, y, cv)
    unionscores, _ = apply_svm(x_union, y, cv)

    params_path = os.path.join(resultsdir,
                                      os.path.splitext(classesfname)[0] + '_params.pkl')
    results_broad_path = os.path.join(resultsdir,
                                      os.path.splitext(classesfname)[0] + '_broadscores.pkl')
    results_narrow_path = os.path.join(resultsdir,
                                      os.path.splitext(classesfname)[0] + '_narrowscores.pkl')
    results_union_path = os.path.join(resultsdir,
                                      os.path.splitext(classesfname)[0] + '_unionscores.pkl')

    pickle.dump(params, open(params_path, 'wb'))
    pickle.dump(broadscores, open(results_broad_path, 'wb'))
    pickle.dump(narrowscores, open(results_narrow_path, 'wb'))
    pickle.dump(unionscores, open(results_union_path, 'wb'))

    # Generate output
    n = len(params)
    bm = broadscores.mean(axis=1).reshape(n, 1)
    bs = broadscores.std(axis=1).reshape(n, 1)
    nm = narrowscores.mean(axis=1).reshape(n, 1)
    ns = narrowscores.std(axis=1).reshape(n, 1)
    um = unionscores.mean(axis=1).reshape(n, 1)
    us = unionscores.std(axis=1).reshape(n, 1)
    summary = np.concatenate([np.array(params), bm, bs, nm, ns, um, us], axis=1)
    h = 'C,deg,gamma,ker,broadmean,broadstd,narrowmean,narrowstd,unionmean,unionstd'
    np.savetxt('/tmp/out.csv', summary, fmt='%s', delimiter=',',
               header=h, comments='')


##########################################################
if __name__ == "__main__":
    main()
