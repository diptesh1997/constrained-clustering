#k-means implementation

#hyperparameters

#k = no of clusters
#penalty weight for must_links
#penalty weight for cannot_links

#selecting initial cluster centres

#cost function that should be minimized

# calculating the cost from point to each of the cluster centres
# Assignment to the cluster with the least weight
# constraints defined as the sum of costs

#constraint filtering: post k-means

#constraints: must link [1,3,5, 4,2,5]
# k_means(5, df, [], must_link, cannot_link)
# assign initial centres
# assigns clusters to -1
# calculates cluster assignment
# prev_clusters = initial clusters
# while (calculated_clusters = prev_clusters)
#   calculate_new_centroids
#   calculate_eucledian_dist
#   penalize constraints
#   new cluster assignment

import math
import numpy as np
from scipy.spatial import distance
from sklearn.utils import shuffle

def k_means(num_clusters, df, init_centres, must_link_penalty, cannot_link_penalty):

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
        centres = choose_initial_centres(num_clusters, df)

    #determine distance of points from the centres and assign clusters
    # df = shuffle(df)
    data = df.to_numpy()
    centroids = centres.to_numpy()
    data = np.hstack([data, np.zeros((len(data),1)), np.ones((len(data),1))])
    #compute the centroids till the cluster assignment remains the same
    fit(data, centroids, num_clusters, must_link_penalty, cannot_link_penalty)

    #print the new csv file after converging

#takes in a dataframe and a center vector and outputs a series with distance values of all points from the vector
def fit(data, centroids, num_clusters, must_link_penalty, cannot_link_penalty):
    iter = 0
    print(centroids)
    while(not np.array_equiv(data[:,5],data[:,4])):
        if(iter!=0):
            data[:,4] = data[:,5]
            centroids = update_centroids(num_clusters, data)
        iter+=1
        dist = []
        for point in data:
            dist_val = []
            for index, center in centroids:
                eucledian_dist = distance.euclidean(point[0:4], center)
                constraint_penalty = penalize(point, index,  )
                dist_val.append(eucledian_dist+constraint_penalty)
            cluster_val = dist_val.index(min(dist_val))
            dist.append(cluster_val)
        data[:,5] = dist
    print(data)
    print('iterations--->',iter)

    #handles case where no point is assigned to a cluster center
    #takes in all data points assigned to individual clusters and asks for clustering again with new centroids
def update_centroids(num_clusters, data):
    #num_columns = len(df.shape[1])
    #make this dynamic: take as many columns as are there in the dataframe
    # clusters_with_no_pts = df[:4].nunique() - num_clusters
    #choose random samples as cluster centres
    # if (clusters_with_no_pts>0):
    #     centers = df.sample(n=clusters_with_no_pts)

    #find mean by cluster value and call eucledian distance again
    centroids = []
    c0 = data[data[:,4]== 0.0]
    c1 = data[data[:,4]== 1.0]
    c2 = data[data[:,4]== 2.0]

    center_0 = np.mean(c0[:,0:4], axis=0)
    center_1 = np.mean(c1[:,0:4], axis=0)
    center_2 = np.mean(c2[:,0:4], axis=0)

    print('center_0-->',center_0)
    print('center_1-->',center_1)
    print('center_2-->',center_2)

    # for i in range(num_clusters):
    #     centroids = center_0

    centroids = [center_0, center_1, center_2]
    return centroids

    #if using dataframes, make binary masks
    # mask_list = []
    # for mask_no in range(num_clusters):
    #     mask_list.append(df['cluster'] == mask_no)

    # df_with_centroids = [] 
    # for mask in mask_list:
    #     df_with_centroids.append(df[mask])

    # centers_df = pd.DataFrame([])
    # for dataframe in df_with_centroids:
    #     centers_df.concat(dataframe.mean(axis=1))

    # print(centers_df)
    # return centers_df

# penalize point for not being assigned to must link peers and being assigned to cannot link peers:
def penalize(point, assumed_pt_cluster, must_link_penalty, cannot_link_penalty):
    postive_sentiment_set = point
    negative_sentiment_set = point
    penalty = 0.0

    #return zero penalty for neutral sentiment documents
    if point['class'] == 0:
        return penalty
        
    elif point['class'] == 1:
        must_link_set = postive_sentiment_set
        cannot_link_set = negative_sentiment_set
    
    else:
        must_link_set = negative_sentiment_set
        cannot_link_set = postive_sentiment_set
    
    #return negative penalty for neutral sentiment documents
    for ml_pt in must_link_set:
        if ml_pt !=point and ml_pt[4]!= -1 and assumed_pt_cluster != ml_pt[4]:
            penalty += must_link_penalty

    #return positive penalty for  sentiment documents
    for cl_pt in cannot_link_set:
        if cl_pt !=point and cl_pt[4] != -1 and assumed_pt_cluster == cl_pt[4]:
            penalty += cannot_link_penalty

    return penalty

