#This major ML Ops line, assumes features have benn extracted
#constraints are applied on run-time while clustering
import sys, getopt

import pandas as pd

from neighbors.neighbours_provider import get_neighbours

n = len(sys.argv)
try:
    ppline=int(sys.argv[1]) #If ppline=1 => Tfidif+sentiment ,If ppline=2 => Word2vec+LDA+sentiment

    neighbours=int(sys.argv[2]) #number of neighbours
    query_doc=sys.argv[3]  #query document id
    n_clusters=int(sys.argv[4]) #number of clusters
    if(ppline==1):
            print("Selected pipeline => "+"Tf_IDF+sentiment")
            doc_path="./feature_extraction/tf_idf_features+sentiment.csv"
    elif (ppline == 2):
            print("Selected pipeline => " + "Word2vec+LDA+sentiment")
            doc_path="./feature_extraction/word2vec_LDA_sentiment.csv"

    else:
        raise Exception("Invalid choice for pipeline")
    print("number of neighbours "+str(neighbours))
    print("Query document " + query_doc)
    print("Number of Clusters " +str(n_clusters ))
    dataframe=pd.read_csv(doc_path,index_col='docno')
    query_point=dataframe[dataframe.index == query_doc].drop("class",axis=1)

    data=get_neighbours(dataframe.drop("class",axis=1),query_point,neighbours)
    final_data = pd.merge(data, dataframe["class"], on="docno")

    print(final_data[["class",'dist']])
except Exception as e:
    print(e)

#Command to run-> python main.py 1 500 'ABC' 400


# try:
#     opts, argv = getopt.getopt(argv, '', ["ppline=", "knn=", "doc_id", "n_clus"])
#     for opt,val in opts:
#         if opt in ("--ppline"):
#             ppline = val
#         if opt in ("--knn"):
#             n_neighbors = val
#         if opt in ("--doc_id"):
#             doc_id = val
#         if opt in ("n_clus"):
#             n_clus = val
# except getopt.error as err:
# 	print (str(err))
#
# #print basic stats on data
# print(f"# samples: {n_samples}; # features {n_features}")


#word2vec pipeline
#vectorized_docs = vectorize(df['text_list'], model=model)
# %%
# Evaluation
# -------------------------------

# * create a pipeline which will scale the data
# * train and time the pipeline fitting;
# * measure the performance of the clustering obtained via different metrics.
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluation(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))