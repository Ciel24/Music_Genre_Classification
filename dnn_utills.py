import csv
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import random
import sys
import io
import os
import glob
import IPython



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


       
    
        
        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
def confusion_matrix(X_data,Y_data,model):
    Y_predicted = []
    for i in range(Y_data.shape[0]):
        Y_predicted.append(np.argmax(Y_data[i,:]))
    Y_predicted = np.array(Y_predicted)
    pred_test = (model.predict(X_data[:]))
    prediction = np.zeros((pred_test.shape[0]))
    for i in range(pred_test.shape[0]):
        prediction[i] = np.argmax(pred_test[i])
    print('           '+ 'E'+ '    ' + 'F' + '    ' +  'G'+ '    ' + 'H'+'   ' + 'M'+'   ' +'S'+'   ' + 'A'+'   ' )
    print(pd.crosstab(Y_predicted, prediction.reshape(X_data.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True))
    plot_confusion_matrix(Y_predicted, prediction)


def accuracy(X,Y,model):
    pred_test = (model.predict(X))
    prediction = np.zeros((pred_test.shape[0]))
    Y_temp_data = np.zeros((pred_test.shape[0]))

    for i in range(pred_test.shape[0]):
        prediction[i] = np.argmax(pred_test[i])
    for i in range(Y.shape[0]):
        Y_temp_data[i] = np.argmax(Y[i])
    print("Accuracy: "  + str(np.mean((prediction == Y_temp_data))*100)+' % ')
    

def model(input_shape,output_size):
    
    X_input = Input(shape = input_shape)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(X_input)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=True)(X_input)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)

    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(output_size)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=X_input, outputs=X)

    ### END CODE HERE ###

    return model
