#!/usr/bin/env python3
"""Plot results from splus
"""

import argparse
import logging
from logging import debug

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score




def autolabel(rects1, rects2, texts):
    """
    Attach a text label above each bar displaying its height
    """
    for rect1, rect2, text in zip(rects1, rects2, texts):
        height = rect1.get_height() + rect2.get_height()
        ax.text(rect1.get_x() + rect1.get_width()/2., 1.01*height,
                text,
                ha='center', va='bottom')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resultscsv', help='Results in .csv format')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    dfall = pd.read_csv(args.resultscsv)
    dfall.r = dfall.r.round()

    df15 = dfall[dfall.r < 16]
    df16 = dfall[dfall.r == 16]
    df17 = dfall[dfall.r == 17]
    df18 = dfall[dfall.r == 18]
    df19 = dfall[dfall.r == 19]
    df20 = dfall[dfall.r > 19]

    n = np.ndarray((6, 3))
    i = 0
    for cl in range(3):
        n[0, cl] = df15[df15.y == cl].shape[0]
        n[1, cl] = df16[df16.y == cl].shape[0]
        n[2, cl] = df17[df17.y == cl].shape[0]
        n[3, cl] = df18[df18.y == cl].shape[0]
        n[4, cl] = df19[df19.y == cl].shape[0]
        n[5, cl] = df20[df20.y == cl].shape[0]

    f1 = np.ndarray((6, 3))
    i = 0
    f1[i, :] = f1_score(df15.y, df15.pred, labels=[0, 1, 2], average=None); i +=1
    f1[i, :] = f1_score(df16.y, df16.pred, labels=[0, 1, 2], average=None); i +=1
    f1[i, :] = f1_score(df17.y, df17.pred, labels=[0, 1, 2], average=None); i +=1
    f1[i, :] = f1_score(df18.y, df18.pred, labels=[0, 1, 2], average=None); i +=1
    f1[i, :] = f1_score(df19.y, df19.pred, labels=[0, 1, 2], average=None); i +=1
    f1[i, :] = f1_score(df20.y, df20.pred, labels=[0, 1, 2], average=None); i +=1
    epsilon = 0.02
    f1[f1 == 0] = epsilon


    #matplotlib setup
    fig, ax1 = plt.subplots()

    colors = ['#7fc97f','#beaed4','#fdc086']
    w = 0.2
    opacity = .8
    i = 0
    rects1b = ax1.bar(np.array(range(15, 21)) + (i-1)*w, f1[:, 0], w,
                      alpha=opacity,
                      color=colors[i],
                      label='Galaxy')
    i += 1
    rects2b = ax1.bar(np.array(range(15, 21)) + (i-1)*w, f1[:, 1], w,
                      alpha=opacity,
                      color=colors[i],                 
                      label='QSO')
    i += 1
    rects3b = ax1.bar(np.array(range(15, 21)) + (i-1)*w, f1[:, 2], w,
                      alpha=opacity,
                      color='tomato',                 
                      label='Star')
    ax1.set_xlabel('Light frequency band')
    ax1.set_ylabel('F1-score')
    labels = list(map(str, range(15, 21)))
    labels[0] = '<16'
    labels[-1] = '>19'
    ax1.set_xticks(range(15, 21), labels)
    ax1.legend()
    ax1.set_title("Classification of " + r"$\bf{catalog}$ using SVM RBF")


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    c = 'tab:blue'
    ec='b'
    ax2.set_ylabel('Number of instances per class', color=c)
    i = 0
    s = 20
    rects1b = ax2.scatter(np.array(range(15, 21)) + (i-1)*w, n[:, 0], s,
                          alpha=opacity,
                          color='tab:blue',
                          label='Galaxy')
    i += 1
    rects2b = ax2.scatter(np.array(range(15, 21)) + (i-1)*w, n[:, 1], s,
                          alpha=opacity,
                          color='tab:blue',
                          label='QSO')
    i += 1
    rects3b = ax2.scatter(np.array(range(15, 21)) + (i-1)*w, n[:, 2], s,
                          alpha=opacity,
                          color='tab:blue',
                          label='Star')
    ax2.tick_params(axis='y', labelcolor=c)

    ax2.set_xticks(range(15, 21), labels)
    fig.tight_layout()
    plt.savefig('/tmp/results.pdf')
    plt.show()
    return

if __name__ == "__main__":
    main()
