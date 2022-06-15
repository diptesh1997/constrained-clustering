from sklearn.neighbors import NearestNeighbors
import pandas as pd

#return the data frame with nearest neighbors of the element
def knn(df, data_point, n_neighbors):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(df)
    return neigh.kneighbors(data_point)
dataset=pd.read_csv("../feature_extraction/Word2Vec.csv",index_col="docno")
Docid = ' LA122989-0001 '
neighbours=50
query_point=dataset[dataset.index==Docid]
print(query_point)
print(knn(dataset, query_point, neighbours))