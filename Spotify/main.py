# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:34:54 2019

@author: agarc
"""
#This file runs various methods defined in the spotifyProjectClasses.py module.
#Methods can be disabled by commenting lines.

#testing change

import numpy as np
import os,sys
import spotifyProjectClasses as spc
import pandas as pd
import time


t0 = time.time()

#setting the working directory to file location
filepath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(filepath)

#opening all user data and music data
#############################################################
user_Filenames = os.listdir("user_data")
music_Filenames = os.listdir("music_data")
#############################################################
data_Base = pd.read_csv("user_data/database.csv",delimiter = ",", dtype = "str")
#############################################################
user_Files = [pd.read_csv("user_data/"+i,delimiter = ",", dtype = "str") for i in user_Filenames if "alpha" in i]
for i in range(len(user_Files)):
    user_Files[i]["spotify_id"] = data_Base["spotify_id"][data_Base["database_id"].isin(user_Files[i]["database_id"])]
user_Data = pd.concat(user_Files).as_matrix()
#############################################################
music_Files = [pd.read_csv("music_data/"+i,delimiter = ",", dtype = "str") for i in music_Filenames]
for i in range(len(music_Files)):
    music_Files[i]["year"] = music_Filenames[i][:-4]
music_Data = pd.concat(music_Files).as_matrix()
#############################################################
t1 = time.time()


#initializing a set of user objects based on the musicFan class
m_Fans = []
#adding songs to songs object, contains all songs
songs = spc.songs(music_Data[np.isfinite(np.float64(music_Data[:,7]))])

for i in range(10):
    #opening user data:
    temp = [[user_Data[x][y+1] for x in range(len(user_Data)) if i == int(user_Data[x][14])] for y in range(16) if y !=13]
    #creating list of user objects:
    m_Fans.append(spc.musicFan(i,temp))
    #users have a variable called cSim which is the output of cosineSimilarity:
    m_Fans[i].getFromData(data_Base)
    m_Fans[i].cosineSimilarity(songs)


t2 = time.time()

print(t1-t0,t2-t1)

#for example:
#mFan 0 has a cSim matrix of shape 109x265119:
#this is because User0 liked 109 songs. Each song has an associated matrix of length 265119 attached to it.
print(m_Fans[0].cSim.shape)
#this matrix has values ranging from 0-1 where 0 is not similar and 1 is identical.
#we can find all songs that are very similar to the suer-liked song by using a mask:
print(songs.name[m_Fans[0].cSim[0]>0.992])   
#this prints out the names of songs whose features are similar to those of song0 in user0
#we can probably improve this by removing features that have little effect on the actual sound. I have tested some of these by comparing the songs on youtube
#and it seems to work decently.
    
 