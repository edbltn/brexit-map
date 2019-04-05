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


class SomTrainer(object):

    def __init__(self, avg_bucket_size: int, epochs: int, 
            docs: List[Document]) -> None:

        self.avg_bucket_size = avg_bucket_size
        self.epochs = epochs
        self.docs = docs
        self.som_size = self.__calc_som_size(docs, avg_bucket_size)
        self.docvecs = self.__unpack_docs(docs)

    def build_som(self):
        vector_size = len(self.docvecs[0])
        som = MiniSom(self.som_size, 
            self.som_size, vector_size, random_seed=42)
        som.train_random(self.docvecs, self.epochs)

        self.som = som

        return self

    @property
    def som_winners(self):
        if not self.som:
            return None

        som_winners = np.empty((len(self.docs), 2), dtype=int)
        for i, doc in enumerate(self.docs):
            winner = self.som.winner(doc.docvec)
            som_winners[i][0] = int(winner[0])
            som_winners[i][1] = int(winner[1])
            
        return som_winners

    @property
    def som_weights(self):
        if not self.som:
            return None

        som_weights_shape = self.som._weights.shape
        som_weights = np.zeros((som_weights_shape[0] * som_weights_shape[1],
                                som_weights_shape[2]))

        for x in range(som_weights_shape[0]):
            for y in range(som_weights_shape[1]):
                index = self.__xy_to_index(x, y, som_weights_shape[1])
                som_weights[index] = self.som._weights[x][y]

        return som_weights

    def build_reduced_som_clusters(self,
            pca_dim: int=300,
            umap_dim: int=5,
            n_clusters: int=100,
            clustering: str="agg"):

        pca_som_weights = self.__pca_som_weights(pca_dim)
        umap_som_weights = self.__umap_som_weights(umap_dim, 
            pca_som_weights)

        reduced_som_clusters = self.__cluster_som(umap_som_weights,
            n_clusters, clustering)

        return reduced_som_clusters

    def build_labels(self, som_clusters):

        labels = []
        som_winners = self.som_winners
        for i in range(som_winners.shape[0]):
            x_val = som_winners[i][0]
            y_val = som_winners[i][1]
            index = self.__xy_to_index(x_val, y_val, self.som_size)
            label = som_clusters[index]
            labels.append(label)

        return labels

    def assign_winners_and_labels(self, labels):
        
        som_winners = self.som_winners
        for i, label in enumerate(labels):
            self.docs[i].label = label
            self.docs[i].grid_point = som_winners[i]

        return self

    def __cluster_som(self, 
        weights,
        n_clusters: int=100, 
        clustering: str="agg"):

        # raise error if clustering isn't "agg" or "kmeans"
        
        if clustering == "agg":
            clusters = AgglomerativeClustering(n_clusters=n_clusters,
                affinity="euclidean", 
                linkage="ward").fit_predict(weights)
        elif clustering == "kmeans":
            clusters = KMeans(n_clusters).fit_predict(weights)
        else:
            clusters = None

        return clusters

    def __umap_som_weights(self, umap_dim, pca_som_weights):
        w = UMAP(n_neighbors=20,
            min_dist=0.1,
            n_components=umap_dim).fit_transform(pca_som_weights)

        return w

    def __pca_som_weights(self, pca_dim):
        w = PCA(n_components=pca_dim).fit_transform(self.som_weights)

        return w


    def __xy_to_index(self, x: int, 
                y: int, size: int) -> int:

        return int(x * size + y)

    def __unpack_docs(self, 
                docs: List[Document]) -> List[np.array]:

        return [doc.docvec for doc in docs]


    def __calc_som_size(self, 
            docs: List[Document], 
            avg_bucket_size: int) -> int:

        som_size = math.sqrt(len(docs) / avg_bucket_size)
        
        return int(round(som_size))
