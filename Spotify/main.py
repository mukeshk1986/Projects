# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:34:54 2019

@author: agarc
"""

#testing gitHub commit

import numpy as np
import os,sys
import spotifyProjectClasses as spc
import pandas as pd
import time


t0 = time.time()

#setting the working directory to file location
filepath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(filepath)

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

m_Fans = []
songs = []

for i in range(10):
    temp = [[user_Data[x][y+1] for x in range(len(user_Data)) if i == int(user_Data[x][14])] for y in range(16) if y !=13]
    m_Fans.append(spc.musicFan(i,temp))
    #m_Fans[i].mean()
    #m_Fans[i].makeScatters()
    #m_Fans[i].makeHistos()

for i in music_Data:
    songs.append(spc.song(i))

t2 = time.time()

print(t1-t0,t2-t1)   
