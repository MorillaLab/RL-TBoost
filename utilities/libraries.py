#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install -U giotto-tda


# In[ ]:


#!pip install -U giotto-tda


# In[ ]:


#!pip install openml


# In[ ]:


print("I am in libs")


# # Import Libraries¶

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prettytable import PrettyTable

from sklearn.preprocessing import RobustScaler, StandardScaler,LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score

#multiinterpolation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import torch
from torch import nn
from torch.utils.data import DataLoader
from gtda.diagrams import Amplitude
import sklearn
from gtda.homology import VietorisRipsPersistence

from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import math
from sklearn.pipeline import make_pipeline, make_union
from gtda.diagrams import PersistenceEntropy
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight


plt.rc('figure', max_open_warning = 0)

import warnings
warnings.filterwarnings("ignore")


# In[ ]:




