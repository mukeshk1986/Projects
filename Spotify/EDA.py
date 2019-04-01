import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df_50 = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\alpha50.csv")
df_75 = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\alpha75.csv")
df_100 = pd.read_csv(r"C:\Users\Mukesh\Documents\Python Scripts\2nd_sem_ML_Stat\spotify_data_for_students_by_ST_ML\spotify_data_for_students\user_data\alpha100.csv")
temp=[df_50,df_75,df_100]
df_final=pd.concat(temp)
print(df_final.shape)
sns.heatmap(df_final.iloc[:,1:14].corr())
plt.show()
