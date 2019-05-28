#!/usr/bin/env python3
"""Assemble the big spectra matrix
python assemble_spectra_matrix.py --spectradir ~anamartinazzo/raw-data/spectra/python assemble_spectra_matrix.py --spectradir ~anamartinazzo/raw-data/spectra/ --filter
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
import logging
from logging import debug
import pickle



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spectradir', required=True, help='Spectra dir, one file per obj')
    parser.add_argument('--filter', required=False, action='store_true',
                        help='Filter by the 16-19 band')
    parser.add_argument('--outdir', required=False, default='/tmp',
                        help='Output dir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG
                        )

    files = sorted(os.listdir(args.spectradir))
    nfiles = len(files)
    spectra = np.ndarray((nfiles, 5500), dtype=float)
    suff = '_1619' if args.filter else ''

    i = 0 
    ids = []
    for f in files:
        if not f.endswith('txt'): continue
        filepath = os.path.join(args.spectradir, f)
        row = np.loadtxt(filepath)
        spectra[i, :] = row
        i += 1
        ids.append(f.replace('.txt', ''))
        debug(i)
        #################################################################
        #if i == 1: break
    spectra = spectra[:i]

    spectra = pd.DataFrame(spectra)
    spectra.columns = spectra.columns.astype(str)
    spectra['id'] = ids

    # Get class from external file
    dr = pd.read_csv('data/dr_early.csv')
    #dr = pd.read_csv('data/dr_early_1619.csv')
    if args.filter: dr = dr[(dr.r>=16) & (dr.r<19)]

    # First filter out non-existing ground truth
    ids_common = set(ids).intersection(set(dr.id))
    spectra = spectra[spectra.id.isin(ids_common)]

    # Secondly get classes
    drfiltered = dr[dr.id.isin(ids)][['id', 'class']]
    spectra = pd.merge(spectra, drfiltered, on='id')
    spectraoutpath = os.path.join(args.outdir, 'spectra' + suff + '.pkl')
    spectra.to_pickle(spectraoutpath)
    debug('Path generated to ' + spectraoutpath)

    x_train, x_test, y_train, y_test = train_test_split(spectra.id, spectra['class'],
                                                        test_size=0.125, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                        test_size=0.143, random_state=1)
    x_train.to_csv(os.path.join(args.outdir, 'train_spectra' + suff + '.txt'),
                   index=False, header=False)
    x_val.to_csv(os.path.join(args.outdir, 'val_spectra' + suff + '.txt'),
                 index=False, header=False)
    x_test.to_csv(os.path.join(args.outdir, 'test_spectra' + suff + '.txt'),
                  index=False, header=False)
    debug('Partitions in  ' + args.outdir)

if __name__ == "__main__":
    main()
