from .doc2vec_trainer import Document
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.decomposition import PCA
from typing import List
from umap import UMAP

import math
import numpy as np
import scipy.cluster.hierarchy as shc


class ClusterTrainer(object):

    def __init__(self, n_clusters: int, 
            docs: List[Document]) -> None:

        self.n_clusters = n_clusters 
        self.docs = docs
        self.docvecs = self.__unpack_docs(docs)
    
    def build_reduced(self,
            docvecs: np.array=None,
            pca_dim: int=300,
            umap_dim: int=5):

        if docvecs is None:
            docvecs = self.docvecs
        
        pca_docvecs = self.__pca_docvecs(pca_dim, docvecs)
        umap_docvecs = self.__umap_docvecs(umap_dim, pca_docvecs)

        return umap_docvecs

    def build_clusters(self,
            docvecs: np.array=None,
            n_clusters: int=100,
            cluster_type: str="agg"):

        if docvecs is None:
            docvecs = self.docvecs

        reduced_clusters = self.__cluster(docvecs, cluster_type)

        return reduced_clusters

    def assign_dims_and_labels(self, docvecs, labels):
        
        for i, (docvec, label) in enumerate(zip(docvecs, labels)):
            self.docs[i].dim_one = docvec[0]
            self.docs[i].dim_two = docvec[1]
            self.docs[i].label = label

        return self

    def __cluster(self, 
        weights,
        cluster_type: str="agg"):

        # raise error if cluster_type isn't "agg" or "kmeans"
        
        if cluster_type == "agg":
            clusters = AgglomerativeClustering(n_clusters=self.n_clusters,
                affinity="euclidean", 
                linkage="ward").fit_predict(weights)
        elif cluster_type == "kmeans":
            clusters = KMeans(self.n_clusters).fit_predict(weights)
        else:
            clusters = None

        return clusters

    def __umap_docvecs(self, umap_dim, pca_docvecs):
        w = UMAP(n_neighbors=20,
            min_dist=0.99,
            n_components=umap_dim).fit_transform(pca_docvecs)

        return w

    def __pca_docvecs(self, pca_dim, docvecs):
        w = PCA(n_components=pca_dim).fit_transform(docvecs)

        return w


    def __xy_to_index(self, x: int, 
                y: int, size: int) -> int:

        return int(x * size + y)

    def __unpack_docs(self, 
                docs: List[Document]) -> List[np.array]:

        return [doc.docvec for doc in docs]
