#!/usr/bin/env python3
""" Tutorial from astropy doc
"""

import argparse
import os
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.utils.data import download_file
from astropy.io import fits

def read_fits1(image_file):
    hdu_list = fits.open(image_file)
    hdu_list.info()

    image_data = hdu_list[0].data

    print(type(image_data))
    print(image_data.shape)
    hdu_list.close()
    return image_data

def read_fits2(image_file):
    image_data = fits.getdata(image_file)
    return image_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()


    horsepath = 'HorseHead.fits'
    if not os.path.exists(horsepath):
        image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True)
        image_filefh = open(horsepath, 'wb')
        pickle.dump(image_file, image_filefh)
        image_filefh.close()
    else:
        image_filefh = open(horsepath, 'rb')
        image_file = pickle.load(image_filefh)

    image_data = read_fits1(image_file)
    #plt.imshow(image_data, cmap='gray')
    #plt.colorbar()
    #plt.show()

    print('Min:', np.min(image_data))
    print('Max:', np.max(image_data))
    print('Mean:', np.mean(image_data))
    print('Stdev:', np.std(image_data))

    print(type(image_data.flatten()))
    nbins = 1000
    #hist = plt.hist(image_data.flatten(), nbins)
    #plt.show()

    plt.imshow(image_data, cmap='gray', norm=LogNorm())
    cbar = plt.colorbar(ticks=[5.e3, 1.e4, 2.e4])
    cbar.ax.set_yticklabels(['5,000', '10,000', '20,000'])
    #plt.show()


    image_list = [ download_file('http://data.astropy.org/tutorials/FITS-images/M13_blue_000'+n+'.fits', cache=True ) \
                  for n in ['1','2','3','4','5'] ]

    image_concat = [ fits.getdata(image) for image in image_list ]
    final_image = np.sum(image_concat, axis=0)

    image_hist = plt.hist(final_image.flatten(), 1000)

    plt.imshow(final_image, cmap='gray', vmin=2.e3, vmax=3.e3)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
