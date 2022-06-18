import math
import numpy as np
from scipy.spatial import distance
from sklearn.utils import shuffle

#K-means
# assign initial centres
# assigns clusters to -1
# calculates cluster assignment
# prev_clusters = initial clusters
# while (calculated_clusters = prev_clusters)
#   calculate_new_centroids
#   calculate_eucledian_dist
#   penalize constraints
#   new cluster assignment
def k_means(num_clusters, df, init_centres, pos_docs, neg_docs, neu_docs, must_link_penalty, cannot_link_penalty):

    #returns a list of k initial centre points for cluster initialization
    def choose_initial_centres(num_clusters, df):
        num_docs = len(df)
        every_x_item = num_docs/num_clusters
        df_centres = df[::math.ceil(every_x_item)]
        return df_centres.iloc[:num_clusters]

    #one time initialization steps    
    #add new columns in dataframe that would contain distance values from the point to the centre
    # df1 = pd.DataFrame(columns=list(map(lambda x: "dist_c"+str(x), range(num_clusters))))
    # df = df.join(df1, how="outer")
    # print(df)

    #choose centre points from data if not already given
    if(len(init_centres)!=num_clusters):
        num_centres = num_clusters/3
        rem_centres = num_clusters%3
        pos_centers = choose_initial_centres(num_centres+rem_centres, pos_docs)
        neg_centers = choose_initial_centres(num_centres, neg_docs)
        neu_centres = choose_initial_centres(num_centres, neu_docs)
        centres = centres.append(pos_centers, neg_centers, neu_centres)
        print(centres)
        
    #determine distance of points from the centres and assign clusters
    # df = shuffle(df)
    data = df.to_numpy()
    col_count = len(data[:][0])
    centroids = centres.to_numpy()
    data = np.hstack([data, np.zeros((len(data),1)), np.ones((len(data),1))])
    #compute the centroids till the cluster assignment remains the same
    fit(data, col_count, centroids, num_clusters, must_link_penalty, cannot_link_penalty)

#takes in a dataframe and a center vector and outputs a series with distance values of all points from the vector
#last 2 columns reserved for cluster comparision (current cluster and prev cluster)
def fit(data, col_count, centroids, num_clusters, must_link_penalty, cannot_link_penalty):
    iter = 0
    pos_docs_loc, neg_docs_loc, neu_docs_loc = extract_sentiment_index()
    while(not np.array_equiv(data[:,col_count+1],data[:,col_count])):
        if(iter!=0):
            data[:,col_count] = data[:,col_count+1]
            data[:,col_count+1] = -1
            centroids = update_centroids(num_clusters, data[:,:col_count],col_count)
        iter+=1
        dist = []
        for point in data:
            dist_val = []
            for index, center in centroids:
                eucledian_dist = distance.euclidean(point[:col_count-1], center)
                penalty_dist = penalize(point, col_count, index, pos_docs_loc, neg_docs_loc, must_link_penalty, cannot_link_penalty)
                dist_val.append(eucledian_dist+penalty_dist)
            cluster_val = dist_val.index(min(dist_val))
            dist.append(cluster_val)
        data[:,col_count+1] = dist
    print('iterations--->',iter)


# penalize point for not being assigned to must link peers and being assigned to cannot link peers:
def penalize(point, col_count, assumed_pt_cluster, pos_docs_loc, neg_docs_loc, must_link_penalty, cannot_link_penalty):
    
    penalty = 0.0

    #return zero penalty for neutral sentiment documents
    if point[col_count-1] == 0:
        return penalty
        
    elif point[col_count-1] == 1:
        must_link_set = pos_docs_loc
        cannot_link_set = neg_docs_loc
    
    else:
        must_link_set = neg_docs_loc
        cannot_link_set = pos_docs_loc
    
    #return negative penalty for neutral sentiment documents
    for ml_pt in must_link_set:
        if np.not_equal(data[ml_pt],point) and data[ml_pt][col_count+1] != -1 and assumed_pt_cluster != data[ml_pt][col_count+1]:
            penalty += must_link_penalty

    #return positive penalty for  sentiment documents
    for cl_pt in cannot_link_set:
        if np.not_equal(data[cl_pt],point) and data[cl_pt][col_count+1] != -1 and assumed_pt_cluster == data[cl_pt][col_count+1]:
            penalty += cannot_link_penalty

    return penalty

#handles case where no point is assigned to a cluster center
def update_centroids(num_clusters, data, col_count):
    #num_columns = len(df.shape[1])
    #make this dynamic: take as many columns as are there in the dataframe
    # clusters_with_no_pts = df[:4].nunique() - num_clusters
    #choose random samples as cluster centres
    # if (clusters_with_no_pts>0):
    #     centers = df.sample(n=clusters_with_no_pts)
    #find mean by cluster value and call eucledian distance again
    
    centroids = []
    data_per_cluster = []
    for clus_no in range(num_clusters):
        data_per_cluster.append(data[data[:,col_count]== float(clus_no)])
    
    for cluster_data in range(len(data_per_cluster)):
        centroids.append(np.mean(cluster_data[:,0:col_count-1], axis=0))
    
    return centroids
