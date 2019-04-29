import numpy as np
import os,sys
import spotifyProjectClasses as spc
import pandas as pd
import time

#setting the working directory to file location
filepath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(filepath)

#opening all user data and music data
print("LOADING FILES")
print("*----------------------------------------------------*")
user_Filenames = os.listdir("user_data")
music_Filenames = os.listdir("music_data")
#----------------------------------------------------------------------------------------------------#
data_Base = pd.read_csv("user_data/database.csv",delimiter = ",", dtype = "str")
#----------------------------------------------------------------------------------------------------#
user_Files = [pd.read_csv("user_data/"+i,delimiter = ",", dtype = "str") for i in user_Filenames if "alpha" in i]
for i in range(len(user_Files)):
    user_Files[i]["spotify_id"] = data_Base["spotify_id"][data_Base["database_id"].isin(user_Files[i]["database_id"])]
user_Data = pd.concat(user_Files).values
#----------------------------------------------------------------------------------------------------#
music_Files = [pd.read_csv("music_data/"+i,delimiter = ",", dtype = "str") for i in music_Filenames]
for i in range(len(music_Files)):
    music_Files[i]["year"] = music_Filenames[i][:-4]
music_Data = pd.concat(music_Files).values
#----------------------------------------------------------------------------------------------------#

#initializing a set of user objects based on the musicFan class
m_Fans = []
songs = spc.songs(music_Data[np.isfinite(np.float64(music_Data[:,7]))])

#This loop creates a list of m_Fan objects.
for i in range(10):
    temp = [[user_Data[x][y+1] for x in range(len(user_Data)) if i == int(user_Data[x][14])] for y in range(16) if y !=13]
    m_Fans.append(spc.musicFan(i,temp))
    #m_Fans[i].makeScatters(songs)
    #m_Fans[i].makeHistos(songs)


for i in range(10):
    m_Fans[i].getFromData(data_Base)

#this test the cSim model for each user.
spc.musicFan.train_Test(m_Fans,KM=True)
#spc.musicFan.train_Test(m_Fans,KNN=True)
#spc.musicFan.train_Test(m_Fans,logistic=True)
#running cSim model and outputting recommendation:
#for i in range(10):
#    m_Fans[i].cosineSimilarity(songs,outputRec=True)





#------------------------------------------------------------------------------# 

