import numpy as np
from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook
output_file("/tmp/color_scatter.html")
import pandas as pd


df = pd.read_csv('/home/dufresne/temp/summary.csv')
source = ColumnDataSource(df)
title = 'Accuracy of classification using Narrow, Broad and All bands.'

tooltips = [
    ("accuracy", "$y"),
    ("ker", "@C"),
    ('degree', '@deg'),    
    ('gamma', '@gamma'),
]

p = figure(tools='pan,hover,wheel_zoom,reset', active_scroll='wheel_zoom', tooltips=tooltips, title=title)
p.circle(x='index', y='narrowmean', source=source,
          fill_color='magenta', fill_alpha=1.0, size=10,
        line_alpha=0, legend='Narrow')
p.circle(x='index', y='broadmean', source=source,
          fill_color='black', fill_alpha=1.0, size=10,
        line_alpha=0, legend='Broad')
p.circle(x='index', y='unionmean', source=source,
          fill_color='orange', fill_alpha=1.0, size=10,
        line_alpha=0, legend='All')

show(p)

