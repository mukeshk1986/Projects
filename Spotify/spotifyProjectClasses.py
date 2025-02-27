import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import os
import collections

class musicFan:
    
    keys = np.array(["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness"\
            ,"liveness","valence","tempo"])
    indices = [0,1,2,3,4,5,6,7,8,9,10]
    
    def __init__(self,ID,tastes):
        
        self.ID = ID
        self.session = np.int_(tastes[11])
        self.dataID = np.array(tastes[13],dtype = "str")
        #self.songID = tastes[14] #this has issues right now, always use dataID
        self.response = np.int_(tastes[12]) == 1 #Mask array of True or False where True is like and False is no like.
        self.tastes = np.array([np.float64(tastes[0]), np.float64(tastes[1]), np.int_(tastes[2])\
                       , np.float64(tastes[3]), np.int_(tastes[4]), np.float64(tastes[5]),\
                       np.float64(tastes[6]), np.float64(tastes[7]), np.float64(tastes[8])\
                       , np.float64(tastes[9]), np.float64(tastes[10])]).T
        self.songNames = []
        self.simSongsNames = []
        self.simSongsID = []
        self.fMask = [True,True,True,True,False,True,True,False,True,False,True]
        #Discarded features: Valence,Instrumentalness
        #self.fMask = [True,True,True,True,True,True,True,True,True,True,True]
        
        print("CREATING musicFan INSTANCE FOR USER:",self.ID,"WITH ITEMxFEATURE MATRIX OF SHAPE:",self.tastes.shape)
        print("*----------------------------------------------------*")
        
        if not os.path.exists("UserOutput"):
            os.makedirs("UserOutput")
        if not os.path.exists("UserOutput/User"+str(self.ID)):
            os.makedirs("UserOutput/User"+str(self.ID))
        
        
    def getFromData(self,dataBase):
        print("GETTING ADDITONAL DATA FROM DB FILE FOR USER:",self.ID)
        print("*----------------------------------------------------*")
        self.songNames = []
        songNames = np.array(dataBase["song_name"])
        dID = np.array(dataBase["database_id"])
        for i in range(len(self.dataID)):
            for x in range(len(songNames)):
                if dID[x] in self.dataID[i]:
                    self.songNames.append(songNames[x])
        self.songNames = np.array(self.songNames)

        
    def mean(self):
        self.mean_Tastes = np.mean(self.tastes, axis = 1)
        self.median_Tastes = np.median(self.tastes, axis = 1)
        self.percent_Positive = np.mean(self.response)
        print("For musicFan "+str(self.ID)+" the mean values are:\n",self.mean_Tastes,"\n","The median values are:\n"\
              ,self.median_Tastes,"\n","Percent positive: " +str(self.percent_Positive)+"\n")
        #return self.mean_Tastes

    def makeScatters(self,songs):
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
            ax1.scatter(songs.features[:,i[0]][::500],songs.features[:,i[1]][::500], color = "red")
            ax1.scatter(self.tastes[:,i[0]][self.response==1],self.tastes[:,i[1]][self.response==1], color = "blue")
            ax1.set_xlabel(self.keys[i[0]])
            ax1.set_ylabel(self.keys[i[1]])
            ax1.set_title(self.keys[i[0]]+"vs."+self.keys[i[1]])
            plt.savefig("UserOutput/User"+str(self.ID)+"/scatters/"+self.keys[i[0]]+"_"+self.keys[i[1]]+".png")
            plt.close("all")
            
        print(combinations)
        print(len(combinations))
        
    def makeHistos(self,songs):
        
        if not os.path.exists("UserOutput/User"+str(self.ID)+"/histograms"):
            os.makedirs("UserOutput/User"+str(self.ID)+"/histograms")
        
        for i in self.indices:
            fig1, ax1 = plt.subplots()
            ax1.grid(linestyle="--")
            #ax1.hist(songs.features[:,i], bins = 20, color = "black",density=True,histtype = "step")#all songs
            ax1.hist(self.tastes[:,i][self.response==0], bins = 20, color = "red",density=True,histtype = "step",label="0",linewidth=2.0)#song with no like
            ax1.hist(self.tastes[:,i][self.response==1], bins = 20, color = "blue",density=True,histtype = "step",label="1",linewidth=2.0)#songs liked
            ax1.set_xlabel(self.keys[i])
            ax1.set_ylabel("normalized count")
            ax1.set_title("Histogram of "+self.keys[i]+" for User: "+str(self.ID))
            ax1.legend()
            plt.savefig("UserOutput/User"+str(self.ID)+"/histograms/"+self.keys[i]+"_histogram"+".png")
            plt.close("all")
            
    def cosineSimilarity(self,songs,outputRec = False):
        a = sklearn.preprocessing.normalize(self.tastes[:,self.fMask][self.response == 1],axis=0)
        print(self.tastes[:,self.fMask][self.response == 1].shape)
        b = sklearn.preprocessing.normalize(songs.features[:,self.fMask],axis=0)
        print(songs.features[:,self.fMask].shape)
        cSim = sklearn.metrics.pairwise.cosine_similarity(a,b)
        self.simSongsNames = np.array([songs.name[i>0.95] for i in cSim])
        self.simSongsID = np.array([songs.ID[i>0.95] for i in cSim])
        if outputRec:
            a = np.concatenate([songs.ID[cSim[i]>0.95] for i in range(len(cSim))])
            b = collections.Counter(a).most_common() #returns a count of values        
            print("Recommended songs for user:",self.ID,".")
            print("ID",b[0:10])
            return b
        print("Cosinesimilarity successful for user:",self.ID,".")
    
    def train_Test(userList,cSim=False,KNN=False,logistic=False,KM=False):
        print("INITIATING MODEL TESTING")
        print("*----------------------------------------------------*")
        if cSim:
            print("For a user, i, the chance of cosineSimilarity recommendation being better than random song picks is:\n")
            for i in userList:
                plus = 0
                size = 0
                for y in range(100):
                    X_train, X_test, y_train, y_test = train_test_split(i.tastes,i.response,test_size = 0.3, random_state=y)
                    a = sklearn.preprocessing.normalize(X_train[:,i.fMask][y_train == 1],axis=0)
                    b = sklearn.preprocessing.normalize(X_test[:,i.fMask],axis=0)
                    labels = np.array([str(x) for x in range(len(b))])
                    cSim = sklearn.metrics.pairwise.cosine_similarity(a,b)
                    c = np.concatenate([labels[cSim[x]>0.985] for x in range(len(cSim))])
                    d = collections.Counter(c).most_common() #returns a count of values
                    if y_test[[int(x[0]) for x in d[0:10]]].sum()/len(y_test[[int(x[0]) for x in d[0:10]]]) > y_test.sum()/len(y_test):
                        plus = plus+1
                        size = size + len(y_test[[int(x[0]) for x in d[0:10]]])
                print("For:",i.ID,":",plus/(y+1))
        if KNN:
            print("For a user, i, the average accuracy of KNN recommendation is:\n")
            for i in userList:
                acc = 0
                for y in range(100):
                    X_train, X_test, y_train, y_test = train_test_split(i.tastes,i.response,test_size = 0.3, random_state=y)
                    X_train = sklearn.preprocessing.normalize(X_train,axis=0)
                    X_test = sklearn.preprocessing.normalize(X_test,axis=0)
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_train[:,i.fMask], y_train)
                    y_Pred = knn.predict(X_test[:,i.fMask])
                    acc = acc + sklearn.metrics.accuracy_score(y_test,y_Pred)
                print("For:",i.ID,":",acc/(y+1))
        if logistic:
            print("For a user, i, the average accuracy of logistic regression recommendation is:\n")
            for i in userList:
                acc = 0
                for y in range(100):
                    X_train, X_test, y_train, y_test = train_test_split(i.tastes,i.response,test_size = 0.3, random_state=y)
                    X_train = sklearn.preprocessing.normalize(X_train,axis=0)
                    X_test = sklearn.preprocessing.normalize(X_test,axis=0)
                    logReg = LogisticRegression(solver='lbfgs')
                    logReg.fit(X_train[:,i.fMask], y_train)
                    y_Pred = logReg.predict(X_test[:,i.fMask])
                    acc = acc + sklearn.metrics.accuracy_score(y_test,y_Pred)
                print("For:",i.ID,":",acc/(y+1))
        if KM:
            print("For a user, i, the average accuracy of logistic regression recommendation is:\n")
            for i in userList:
                acc = 0
                for y in range(10):
                    X_train, X_test, y_train, y_test = train_test_split(i.tastes,i.response,test_size = 0.3, random_state=y)
                    X_train = sklearn.preprocessing.normalize(X_train,axis=0)
                    X_test = sklearn.preprocessing.normalize(X_test,axis=0)
                    kMeans = KMeans(n_clusters = 2)
                    kMeans.fit(X_train[:,i.fMask], y_train)
                    y_Pred = kMeans.predict(X_test[:,i.fMask])
                    acc = acc + sklearn.metrics.accuracy_score(y_test,y_Pred)
                print("For:",i.ID,":",acc/(y+1))
                
        print("*----------------------------------------------------*")
        print("MODEL TESTING COMPLETE")
        
            
        
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
        
        print("CREATING songs INSTANCE WITH ITEMxFEATURE MATRIX OF SHAPE:",self.features.shape)
        print("*----------------------------------------------------*")