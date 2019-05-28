#!/usr/bin/env python3
"""Assemble catalog matrix, assuming a csv is provided with all fields
python assemble_catalog_matrix.py --fullcsv data/dr1_full.csv --cols data/cols_dr1.txt --outpkl /tmp/out.pkl
"""


import argparse
import os
from os.path import join as ojoin
import time
import logging
from logging import debug
import pickle
import numpy as np

import pandas as pd


##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fullcsv', required=True, help='Path to the full csv file')
    parser.add_argument('--cols', required=True, help='Path to the columns names')
    parser.add_argument('--trainids', required=False, default='data/train.txt',
                        help='Train list path')
    parser.add_argument('--valids', required=False, default='data/val.txt',
                        help='Validation list path')
    parser.add_argument('--testids', required=False, default='data/test.txt',
                        help='Test list path')
    parser.add_argument('--filter', required=False, action='store_true',
                        help='Filter by the 16-19 band')
    parser.add_argument('--outdir', required=False, default='/tmp', help='Output dir')
    args = parser.parse_args()

    alldata = pd.read_csv(args.fullcsv)
    cols = np.loadtxt(args.cols, dtype=str)
    suff = '_1619' if args.filter else ''

    trainids = set(np.loadtxt(args.trainids, dtype=str))
    valids = set(np.loadtxt(args.valids, dtype=str))
    testids = set(np.loadtxt(args.testids, dtype=str))


    if args.filter:
        alldata = alldata[(alldata.r>=16) & (alldata.r<19)]

    debug(len(trainids))
    debug(len(valids))
    debug(len(testids))

    allids = set(alldata.id)
    trainids = np.array(list(trainids.intersection(allids)))
    valids = np.array(list(valids.intersection(allids)))
    testids = np.array(list(testids.intersection(allids)))

    debug(trainids.shape)
    debug(valids.shape)
    debug(testids.shape)

    np.savetxt(os.path.join(args.outdir, 'train_dr1' + suff + '.txt'), trainids,
               fmt='%s')
    np.savetxt(os.path.join(args.outdir, 'val_dr1' + suff + '.txt'), valids,
               fmt='%s')
    np.savetxt(os.path.join(args.outdir, 'test_dr1' + suff + '.txt'), testids,
               fmt='%s')
    
    features = alldata[list(cols)+['id']+['class']]
    features.to_pickle(os.path.join(args.outdir, 'dr1' + suff + '.pkl'))
    return

##########################################################
if __name__ == "__main__":
    main()
