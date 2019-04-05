from grid.grouping import Grouping 
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from trainers.doc2vec_trainer import Doc2VecTrainer
from trainers.cluster_trainer import ClusterTrainer
from time import time
import csv

def corpus_record_to_dict(row):
    record_dict = dict(index=row.get("index"),
        title=row.get("title"),
        article_text=row.get("article_text"),
        docvec=row.get("docvec"))
    return record_dict

def csv_to_list(csvfilename, field=-1):
    rows = []
    with open(csvfilename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row[field])
    return rows

def print_terms(name, most_unique_terms, tfidf_scores, limit=10):

    print('{} top terms:'.format(name))
    i = 0
    for term, frequency in most_unique_terms.items():
        print('{} Frequency: {}'.format(term, frequency))
        i += 1
        if i > limit:
            break

    i = 0
    for term, tfidf in tfidf_scores.items():
        print('{} TF*IDF score: {}'.format(term, tfidf))
        i += 1
        if i > limit:
            break

if __name__ == "__main__":

    # Train Docvecs
    doc2vec = Doc2VecTrainer(epochs=1,
                             min_count=10,
                             vector_size=300,
                             window=15)
    print("Building corpus...")
    t0 = time()
    doc2vec.get_corpus('../data/20190219_brexit_articles.csv') 
    #doc2vec.add_corpus('../data/20190212_brexit_articles.csv')
    #doc2vec.add_corpus('../data/20190205_brexit_articles.csv')
    #doc2vec.add_corpus('../data/20190129_brexit_articles.csv')

    print("Training Doc2Vec...")
    t1 = time()
    doc2vec.build_model()
    doc2vec.add_docvecs()

    # Build SOM
    print("Building Clustering...")
    t2 = time()
    ct = ClusterTrainer(100, doc2vec.corpus)
    reduced_docvecs_five_dims = ct.build_reduced(pca_dim = 50, umap_dim = 5)
    labels = ct.build_clusters(reduced_docvecs_five_dims)
    reduced_docvecs_two_dims = ct.build_reduced(docvecs = reduced_docvecs_five_dims,
                                                pca_dim = 5,umap_dim = 2)
    ct.assign_dims_and_labels(reduced_docvecs_two_dims, labels)
    docs = ct.docs

    # Build grid
    t3 = time()
    filename = '../data/stopwords-en.csv'
    stoplist = csv_to_list(filename)
    grid = Grouping(docs, stoplist)

    t4 = time()
    print(f'Build corpus time: {t1-t0:.2f} seconds')
    print(f'Train Doc2Vec time: {t2-t1:.2f} seconds')
    print(f'Build SOM time: {t3-t2:.2f} seconds')
    print(f'Build Grid time: {t4-t3:.2f} seconds')

    # Get SOM stats
    # gp = grid.grid_points[(1, 3)]
    # print(gp.top_words[:10], gp.top_bigrams[:10], gp.top_trigrams[:10])
