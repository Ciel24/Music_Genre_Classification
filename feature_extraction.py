import numpy as np 
import librosa 
import os, os.path, errno
import librosa.display 
import datetime 
import json 
import csv 
from timeit import default_timer as timer 
import argparse


M = 1024 
N = 1024 
H = 512 
fs = 22050 
mfccs = []
delta_mfccs = []
delta_delta_mfccs = []





def extract_mfcc():
    file_categories = ['classical','blues','country','disco','hiphop','jazz','metal','pop','reggae','rock']

    for i in range(len(file_categories)):
        for k in range(0,100):
            if (k<10):
                local_file_path = os.getcwd()+'/genres/'+ file_categories[i]+'/'+file_categories[i]+'.0000'+str(k)+'.au'
            else:
                local_file_path = os.getcwd()+'/genres/'+ file_categories[i]+'/'+file_categories[i]+'.000'+str(k)+'.au'

            x,_ = librosa.load(local_file_path,sr = 22050)
            librosa.feature.melspectrogram(x,sr=22050,n_fft=N,hop_length=H)
            S = librosa.feature.melspectrogram(x, sr=22050, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc= librosa.feature.mfcc(S=log_S, n_mfcc=13)
            delta_mfcc  = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            mfccs = np.array(mfcc)
            delta_mfccs = np.array(delta_mfcc)
            delta_delta_mfcc = np.array(delta2_mfcc)

            try:
                path = os.getcwd()+"/Features/MFCC_13_Delta_13_Delta_Delta_13/"+file_categories[i]+"_features/"+str(k)
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            np.savetxt(path+"/mfccs.csv",mfccs.T, delimiter=",")
            np.savetxt(path+"/delta_mfccs.csv",delta_mfccs.T, delimiter=",")
            np.savetxt(path+"/delta_delta_mfcc.csv",delta_delta_mfcc.T, delimiter=",")
            print ('\n')
            print ("MFCC extracted successfully from "+file_categories[i]+'_'+ '('+str(k)+')')
            print ("Delta_MFCC extracted successfully from "+file_categories[i]+'_'+ '('+str(k)+')')
            print ("Delta_Delta_MFCC extracted successfully from "+file_categories[i]+'_'+ '('+str(k)+')')
            print ('\n')
            print ('---------------------------------------------------------------------')



if __name__ == '__main__':
    extract_mfcc()







        


