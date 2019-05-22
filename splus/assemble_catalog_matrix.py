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
    parser.add_argument('--outpkl', required=True, help='Pickle output path')
    args = parser.parse_args()

    alldata = pd.read_csv(args.fullcsv)
    cols = np.loadtxt(args.cols, dtype=str)
    features = alldata[list(cols)+['id']+['class']]
    features.to_pickle(args.outpkl)
    return

##########################################################
if __name__ == "__main__":
    main()
