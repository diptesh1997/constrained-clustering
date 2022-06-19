#This major ML Ops line, assumes features have benn extracted
#constraints are applied on run-time while clustering
import sys, getopt
import traceback

import pandas as pd

from clustering.test_again import k_means
from neighbors.neighbours_provider import get_neighbours
n = len(sys.argv)
try:
    ppline=int(sys.argv[1]) #If ppline=1 => Tfidif+sentiment ,If ppline=2 => Word2vec+LDA+sentiment

    neighbours=int(sys.argv[2]) #number of neighbours
    query_doc=sys.argv[3]  #query document id
    n_clusters=int(sys.argv[4]) #number of clusters
    if(ppline==1):
            print("Selected pipeline => "+"Tf_IDF+sentiment")
            doc_path="../data/tf_idf_features+sentiment.csv"
    elif (ppline == 2):
            print("Selected pipeline => " + "Word2vec+LDA+sentiment")
            doc_path="../data/word2vec_LDA_sentiment.csv"

    else:
        raise Exception("Invalid choice for pipeline")
    print("number of neighbours "+str(neighbours))
    print("Query document " + query_doc)
    print("Number of Clusters " +str(n_clusters ))
    dataframe=pd.read_csv(doc_path,index_col='docno')
    query_point=dataframe[dataframe.index == query_doc].drop("class",axis=1)

    data=get_neighbours(dataframe.drop("class",axis=1),query_point,neighbours)
    final_data = pd.merge(data, dataframe["class"], on="docno")
    final_data=final_data.drop('dist',axis=1)
    pos_doc_df = final_data[final_data["class"] == 1]

    neg_doc_df = final_data[final_data["class"] == -1]
    neu_doc_df = final_data[final_data["class"] == 0]
    keyphrase_df = pd.read_csv("../data/keyphrase_docno_added.csv", index_col='docno')

    result=k_means(n_clusters, final_data, keyphrase_df, [], pos_doc_df, neg_doc_df, neu_doc_df, -0.02, 0.1)
    result['docno']=final_data.index
    result=result.set_index('docno')
    from datetime import datetime

    now = datetime.now()

    print("now =", now)

    dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
    print("date and time =", dt_string)
    filename='./result '+str(dt_string)+'.csv'
    print(filename)
    result.to_csv(filename,index=True)

except Exception as e:
    traceback.print_exc(e)

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
