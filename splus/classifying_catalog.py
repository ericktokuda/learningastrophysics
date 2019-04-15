#!/usr/bin/env python3
"""Read and classify among three objects

"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()
    #pass
    catalogpath = '/foo'
    x = pd.read_csv(catalogpath, sep='  ', header=191, engine='python')
    print(x.head())
    print(x.describe())
    print(x.nrows())


if __name__ == "__main__":
    main()

