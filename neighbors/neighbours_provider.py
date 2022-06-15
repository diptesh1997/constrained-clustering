
import pandas as pd
import numpy as np
from scipy.spatial import distance
def knn(df, data_point, n_neighbors):

    distance_list = distance.cdist(df, data_point, metric='euclidean')
    df['dist'] = distance_list
    sorted_distance = df.sort_values('dist', ascending=True)
    nearest_neighbours = sorted_distance.iloc[:n_neighbors+1]
    return (nearest_neighbours)

df=pd.read_csv('../feature_extraction/Word2Vec.csv',index_col='docno')
distance_frame=df

Docid = ' LA122989-0001 '
neighbours=50
query_point = df[df.index == Docid]

print(knn(df,query_point,neighbours))
