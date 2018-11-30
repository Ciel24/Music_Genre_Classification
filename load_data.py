import numpy as np 
import random
import sys
import io
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
import glob
import IPython


path = os.getcwd()+'/Features/MFCC_13_Delta_13_Delta_Delta_13/'
classical = 100
blues = 100
country = 100
disco = 100
hiphop = 100
jazz = 100 
metal = 100 
pop = 100 
reggae = 100 
rock = 100 

Tx = 1290
m = classical + blues +country+ disco + hiphop + jazz + metal + pop + reggae + rock 
num_features = 41 
X_data = np.zeros((m,Tx,num_features))
Y_data = np.zeros((m))
group_names = ['classical','blues','country','disco','hiphop','jazz','metal','pop','reggae','rock']
group_label = {'classical':0,'blues':1,'country':2,'disco':3,'hiphop':4,'jazz':5,'metal':6,'pop':7,'reggae':8,'rock':9}
group_size = {'classical':classical,'blues':blues,'country':country,'disco':disco,'hiphop':hiphop,'jazz':jazz,'metal':metal,'pop':pop,'reggae':reggae,'rock':rock}
group_features_not_exist = {'classical':[],'blues':[],'country':[],'disco':[],'hiphop':[],'jazz':[],'metal':[],'pop':[],'reggae':[],'rock':[]}
group_number = {'classical':0,'blues':0,'country':0,'disco':0,'hiphop':0,'jazz':0,'metal':0,'pop':0,'reggae':0,'rock':0}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def load_data(classical=100,blues=100,country=100,disco=100,hiphop=100,jazz=100,metal=100,pop=100,reggae=100,rock=100,Tx=1290):
    i = 0
    file_number = 1
    for group in group_names:
        i = 0
        while(i <= group_size[group]):
            try:
                temp = np.genfromtxt(path+ str(group)+'_features/'+str(i)+'/mfccs.csv', dtype=float, delimiter=',')
                temp_delta = np.genfromtxt(path+ str(group)+'_features/'+str(i)+'/delta_mfccs.csv', dtype=float, delimiter=',')
                temp_delta_delta = np.genfromtxt(path+ str(group)+'_features/'+str(i)+'/delta_delta_mfcc.csv', dtype=float, delimiter=',')

            except IOError:
                print (" Reading Error in " +group +' '+ str(i))
                i = i + 1 
                continue
            try:
                X_data[file_number-1,:,0:13] = temp[:Tx,:]
                X_data[file_number-1,:,13:26] = temp_delta[:Tx,:]
                X_data[file_number-1,:,26:39] = temp_delta_delta[:Tx,:]
                X_data[file_number-1,:,39] = i
                X_data[file_number-1,:,40] = group_label[group]
                Y_data[file_number-1] = group_label[group]
                group_number[group] = group_number[group] + 1 
            except:
                group_features_not_exist[group].append(i)
                i = i+1
                continue 
            file_number = file_number + 1 
            i = i + 1
        print ("------------------------------------------")
        print (group_features_not_exist[group])
        print (group+' '+" is done with file number : "+ str(file_number-1)+ " and i = "+ str(i))
        print ( group+ " number is : "+ str(group_number[group]))
        print ("")
    X_train = X_data[:file_number-1,:,:]
    Y = Y_data[:file_number-1]
    temp = np.array([ int(i)for i in Y])
    Y_train = convert_to_one_hot(temp, C = 10)
    np.savez(os.getcwd()+'/npz_data/Data.npz', X_data=X_train, Y_data=Y_train)
    return X_train,Y_train
