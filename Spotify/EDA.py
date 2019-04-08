import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
#merging 3 different distributed files and creating a single dataframe
df_50 = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\alpha50.csv")
df_75 = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\alpha75.csv")
df_100 = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\alpha100.csv")
temp=[df_50,df_75,df_100]
df_user_songs=pd.concat(temp)
#creating dataframe for database files
df_database = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\database.csv")
print(df_user_songs.shape)
print(df_database.shape)
df_database.describe()
sns.heatmap(df_user_songs.iloc[:,1:14].corr())
plt.show()

# # Custom Color Palette
green_red = ['#58D68D','#EF4846']
palette = sns.color_palette(green_red)
sns.set_palette(palette)
sns.set_style("white")
_ih[-5:] # to get the last 5 deleted cells


positive_tempo = df_user_songs[df_user_songs['user_response']== 1]['tempo']
negative_tempo= df_user_songs[df_user_songs['user_response']== 0]['tempo']

fig = plt.figure(figsize = (9,5))
plt.title("Song Tempo Like / Dislike Distribution")
positive_tempo.hist(alpha=0.7, bins = 30, label="Positive Tempo" )
negative_tempo.hist(alpha=0.7, bins = 30, label="Negative Tempo")
plt.legend(loc="upper left")
positive_tempo.head(10)
negative_tempo.head(10)
#bar chart
ax = df_user_songs["user_response"].value_counts().plot(kind='bar')
plt.show()
grid_fig = plt.figure(figsize = (15,15))
# Trying to find some pattern 
positive_tempo = df_user_songs[df_user_songs['user_response']== 1]['tempo']
negative_tempo = df_user_songs[df_user_songs['user_response']== 0]['tempo']

positive_dance = df_user_songs[df_user_songs['user_response']== 1]['danceability']
negative_dance = df_user_songs[df_user_songs['user_response']== 0]['danceability']

positive_duration_ms = df_user_songs[df_user_songs['user_response']== 1]['duration_ms']
negative_duration_ms = df_user_songs[df_user_songs['user_response']== 0]['duration_ms']

positive_loudness = df_user_songs[df_user_songs['user_response']== 1]['loudness']
negative_loudness = df_user_songs[df_user_songs['user_response']== 0]['loudness']

positive_speechiness = df_user_songs[df_user_songs['user_response']== 1]['speechiness']
negative_speechiness = df_user_songs[df_user_songs['user_response']== 0]['speechiness']

positive_valence = df_user_songs[df_user_songs['user_response']== 1]['valence']
negative_valence = df_user_songs[df_user_songs['user_response']== 0]['valence']

positive_energy = df_user_songs[df_user_songs['user_response']== 1]['energy']
negative_energy = df_user_songs[df_user_songs['user_response']== 0]['energy']

positive_acousticness = df_user_songs[df_user_songs['user_response']== 1]['acousticness']
negative_acousticness = df_user_songs[df_user_songs['user_response']== 0]['acousticness']

positive_key = df_user_songs[df_user_songs['user_response']== 1]['key']
negative_key = df_user_songs[df_user_songs['user_response']== 0]['key']

positive_instrumentalness = df_user_songs[df_user_songs['user_response']== 1]['instrumentalness']
negative_instrumentalness = df_user_songs[df_user_songs['user_response']== 0]['instrumentalness']
#Train Test Split
train,test= train_test_split(df_user_songs,test_size=0.20)
print("Training_size = {} ; Test_size = {}" .format(len(train),len(test)))
print("shape of training dataset",train.shape)
print("Shape of test dataset", test.shape)


# Danceability
dan = grid_fig.add_subplot(331) # 3*3 grid at location 1 
dan.set_xlabel('Danceability')
dan.set_ylabel('Count')
dan.set_title('Song Danceability Like Distribution')
positive_dance.hist(alpha=0.5, bins=30)
dan1 = grid_fig.add_subplot(331)
negative_dance.hist(alpha=0.5, bins=30)



# Loudness
lou = grid_fig.add_subplot(333) # 3*3 grid at location 3 
lou.set_xlabel('Loudness')
lou.set_ylabel('Count')
lou.set_title('Song Loudness Like Distribution')
positive_loudness.hist(alpha=0.5, bins=30)
lou1 = grid_fig.add_subplot(333)
negative_loudness.hist(alpha=0.5, bins=30)


# Speechiness
spe = grid_fig.add_subplot(334) # 3*3 grid at location 4
spe.set_xlabel('Speechiness')
spe.set_ylabel('Count')
spe.set_title('Song Speechiness Like Distribution')
positive_speechiness.hist(alpha=0.5, bins=30)
spe1 = grid_fig.add_subplot(334)
negative_speechiness.hist(alpha=0.5, bins=30)


# Valence
val = grid_fig.add_subplot(335) # 3*3 grid at location 5
val.set_xlabel('Valence')
val.set_ylabel('Count')
val.set_title('Song Valence Like Distribution')
positive_valence.hist(alpha=0.5, bins=30)
val1 = grid_fig.add_subplot(335)
negative_valence.hist(alpha=0.5, bins=30)


# Energy
ene = grid_fig.add_subplot(336) # 3*3 grid at location 6 
ene.set_xlabel('Energy')
ene.set_ylabel('Count')
ene.set_title('Song Energy Like Distribution')
positive_energy.hist(alpha=0.5, bins=30)
ene1 = grid_fig.add_subplot(336)
negative_energy.hist(alpha=0.5, bins=30)


# Acousticness
aco = grid_fig.add_subplot(337) # 3*3 grid at location 7 
aco.set_xlabel('Acousticness')
aco.set_ylabel('Count')
aco.set_title('Song Acousticness Like Distribution')
positive_acousticness.hist(alpha=0.5, bins=30)
aco1 = grid_fig.add_subplot(337)
negative_acousticness.hist(alpha=0.5, bins=30)


# Key
key = grid_fig.add_subplot(338) # 3*3 grid at location 8 
key.set_xlabel('Key')
key.set_ylabel('Count')
key.set_title('Song Key Like Distribution')
positive_key.hist(alpha=0.5, bins=30)
key1 = grid_fig.add_subplot(338)
negative_key.hist(alpha=0.5, bins=30)


# Instrumentalness
ins = grid_fig.add_subplot(339) # 3*3 grid at location 9 
ins.set_xlabel('Instrumentalness')
ins.set_ylabel('Count')
ins.set_title('Song Instrumentalness Like Distribution')
positive_instrumentalness.hist(alpha=0.5, bins=30)
ins1 = grid_fig.add_subplot(339)
negative_instrumentalness.hist(alpha=0.5, bins=30)

#feature extraction
c = DecisionTreeClassifier(min_samples_split=100)
features  = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness"]
x_train = train[features]
y_train = train["user_response"]

x_test = test[features]
y_test = test["user_response"]

##
dt = c.fit(x_train,y_train)
def show_tree(tree,features,path):
    f = io.StringIO()
    export_graphviz(tree, out_file=f , feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = misc.imread(os.path)
    plt.rcParams["figure.figsize"]= (20,20)
    plt.imshow(img)
