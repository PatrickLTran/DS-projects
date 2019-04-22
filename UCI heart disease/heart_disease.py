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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.tree import graphviz
import graphviz
data = pd.read_csv('heart.csv')

data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['cp'][data['cp'] == 0] = 'typical angina'
data['cp'][data['cp'] == 1] = 'atypical angina'
data['cp'][data['cp'] == 2] = 'non-anginal pain'
data['cp'][data['cp'] == 3] = 'asymptomatic'

data['fbs'][data['fbs'] == 0] = 'lower than 120mg/ml'
data['fbs'][data['fbs'] == 1] = 'greater than 120mg/ml'

data['restecg'][data['restecg'] == 0] = 'normal'
data['restecg'][data['restecg'] == 1] = 'ST-T wave abnormality'
data['restecg'][data['restecg'] == 2] = 'left ventricular hypertrophy'

data['exang'][data['exang'] == 0] = 'no'
data['exang'][data['exang'] == 1] = 'yes'

data['slope'][data['slope'] == 0] = 'upsloping'
data['slope'][data['slope'] == 1] = 'flat'
data['slope'][data['slope'] == 2] = 'downsloping'

data['thal'][data['thal'] == 0] = 'normal'
data['thal'][data['thal'] == 1] = 'fixed defect'
data['thal'][data['thal'] == 2] = 'reversable defect'


train_dummies = pd.get_dummies(data,drop_first=True)

#split data
X_train, X_test, y_train, y_test = train_test_split(train_dummies.drop('target',1),
                                                    train_dummies['target'],
                                                    test_size=0.2, random_state=10)
####################################################
###########random forest classifier#################
####################################################
rf_model = RandomForestClassifier(max_depth=5)
rf_model.fit(X_train,y_train)

estimator = rf_model.estimators_[1]
feature_names= list(X_train.columns)

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


#tree viewer
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')






from sklearn.preprocessing import StandardScaler
X = train_array.values()
X_std = StandardScaler().fit_transform(X)

#correlation matrix
plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()

temp = (data.groupby(['target']))['cp'].value_counts(normalize=True)\
.mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "cp", data = temp)\
.set_title("Chest Pain vs Heart Disease")




