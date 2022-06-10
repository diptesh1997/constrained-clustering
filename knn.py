from sklearn.neighbors import NearestNeighbors

#return the data frame with nearest neighbors of the element
def knn(df, data_point, n_neighbors):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(df)
    return neigh.kneighbors(data_point)