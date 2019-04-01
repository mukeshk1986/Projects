# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:57:40 2019
@author: agarc
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.preprocessing
import os

class musicFan:
    
    keys = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness"\
            ,"liveness","valence","tempo"]
    indices = [0,1,2,3,4,5,6,7,8,9,10]
    
    def __init__(self,ID,tastes):
        
        self.ID = ID
        self.session = np.int_(tastes[11])
        self.dataID = np.array(tastes[13],dtype = "str")
        self.songID = tastes[14]
        self.response = np.int_(tastes[12])
        self.tastes = np.array([np.float64(tastes[0]), np.float64(tastes[1]), np.int_(tastes[2])\
                       , np.float64(tastes[3]), np.int_(tastes[4]), np.float64(tastes[5]),\
                       np.float64(tastes[6]), np.float64(tastes[7]), np.float64(tastes[8])\
                       , np.float64(tastes[9]), np.float64(tastes[10])]).T
        
        if not os.path.exists("UserOutput"):
            os.makedirs("UserOutput")
        if not os.path.exists("UserOutput/User"+str(self.ID)):
            os.makedirs("UserOutput/User"+str(self.ID))
        
  
    def mean(self):
        self.mean_Tastes = np.mean(self.tastes, axis = 1)
        self.median_Tastes = np.median(self.tastes, axis = 1)
        self.percent_Positive = np.mean(self.response)
        print("For musicFan "+str(self.ID)+" the mean values are:\n",self.mean_Tastes,"\n","The median values are:\n"\
              ,self.median_Tastes,"\n","Percent positive: " +str(self.percent_Positive)+"\n")
        #return self.mean_Tastes

    def makeScatters(self):
        num_Comb = 55
        
        if not os.path.exists("UserOutput/User"+str(self.ID)+"/scatters"):
            os.makedirs("UserOutput/User"+str(self.ID)+"/scatters")
        
        combinations = []
        
        for i in self.indices:
            for j in self.indices:
                if [i,j] not in combinations and [j,i] not in combinations and i != j:
                    combinations.append([i,j])
        for i in combinations:
            fig1, ax1 = plt.subplots()
            ax1.scatter(self.tastes[i[0]][np.isin(self.response,0)],self.tastes[i[1]][np.isin(self.response,0)], color = "red")
            ax1.scatter(self.tastes[i[0]][np.isin(self.response,1)],self.tastes[i[1]][np.isin(self.response,1)], color = "blue")
            ax1.set_xlabel(self.keys[i[0]])
            ax1.set_ylabel(self.keys[i[1]])
            ax1.set_title(self.keys[i[0]]+"vs."+self.keys[i[1]])
            plt.savefig("UserOutput/User"+str(self.ID)+"/scatters/"+self.keys[i[0]]+"_"+self.keys[i[1]]+".png")
            plt.close("all")
        print(combinations)
        print(len(combinations))
        
    def makeHistos(self):
        
        if not os.path.exists("UserOutput/User"+str(self.ID)+"/histograms"):
            os.makedirs("UserOutput/User"+str(self.ID)+"/histograms")
        
        for i in self.indices:
            fig1, ax1 = plt.subplots()
            ax1.hist(self.tastes[i], bins = 20, color = "green")
            ax1.hist(self.tastes[i][np.isin(self.response,0)], bins = 20, color = "red")
            ax1.hist(self.tastes[i][np.isin(self.response,1)], bins = 20, color = "blue")
            ax1.set_xlabel(self.keys[i])
            ax1.set_ylabel("count")
            ax1.set_title("Histogram of "+self.keys[i]+" for user: "+str(self.ID))
            plt.savefig("UserOutput/User"+str(self.ID)+"/histograms/"+self.keys[i]+"_histogram"+".png")
            plt.close("all")
            
    def cosineSimilarity(self,songs):
        a = sklearn.preprocessing.normalize(self.tastes[self.response == 1],axis=0)
        b = sklearn.preprocessing.normalize(songs.features,axis=0)
        self.cSim = sklearn.metrics.pairwise.cosine_similarity(a,b)
        print("Cosinesimilarity successful for user:",self.ID,".")
        
            
        
class songs:

    keys = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness"\
            ,"liveness","valence","tempo"]
    
    def __init__(self,features):
        
        self.ID = features[:,18]
        self.name = features[:,19]
        self.albumID = features[:,2]
        self.albumName = features[:,3]
        self.artistID = features[:,5]
        self.artistName = features[:,6]
        self.year = features[:,26]
        self.features = np.array([np.float64(features[:,7]), np.float64(features[:,10]), np.float64(features[:,12]), np.float64(features[:,14]),\
                                    np.int_(features[:,15]), np.float64(features[:,20]), np.float64(features[:,1]),\
                                    np.float64(features[:,11]),np.float64(features[:,13]),np.float64(features[:,25]),np.float64(features[:,21])]).T
      
    def recommend(self,user):
        
        a = np.concatenate([self.ID[user.cSim[i]>0.95] for i in range(len(user.cSim))])
                
        print(a.shape)
        #return a    
        return np.unique(a, return_counts = True)