# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:07:39 2019

@author: Patrick
"""

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

import plotly.offline as py
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = pd.read_csv('heart.csv')

target = data['target']
train = data.drop('target',axis=1)

train_array = train.values
