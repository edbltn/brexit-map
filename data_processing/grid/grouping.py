from six import iteritems
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
import string
import csv
import pickle
import numpy as np
from collections import defaultdict
from typing import Tuple

# TODO: Count / analyze linked entities in each cluster 

class Group(object):

    def __init__(self, group, docs, text_field, top_words, top_bigrams, top_trigrams):
        self.group = group
        self.docs = docs
        self.text = '\n\n'.join([getattr(doc, text_field) for doc in docs])
        self.top_words = top_words
        self.top_bigrams = top_bigrams
        self.top_trigrams = top_trigrams

    @property
    def term_frequencies(self):
        tokens = simple_preprocess(self.text)
        vectorizer = CountVectorizer(ngram_range=(1,1))
        X = vectorizer.fit_transform(tokens).toarray()
        term_frequencies = {}
        for term in vectorizer.vocabulary_:
            freq = X[:,vectorizer.vocabulary_[term]].sum()
            term_frequencies[term] = freq
        sorted_tf = sorted(term_frequencies.items(),
                key=lambda x: x[1], reverse=True)
        tf_dict = {tf[0]: tf[1] for tf in sorted_tf}
        return tf_dict

    @property
    def top_hundred_frequencies(self):
        return self.term_frequencies[:100]

    @property
    def avg_word_count(self):
        wc = [doc.word_count for doc in self.docs]
        return sum(wc) / len(wc)

    def build_most_unique_terms(self, other):
        other_text = '\n'.join([o.text for o in other])
        other_tokens = simple_preprocess(other_text)
        other_set = set(other_tokens)

        tokens = simple_preprocess(self.text)

        unique_terms = list(set(tokens) - other_set)
        unique_terms_dict = {ut: self.term_frequencies[ut]
                for ut in unique_terms}

        return unique_terms_dict

class Cluster(Group):

    def __init__(self, cluster: int, docs, text_field, top_words, top_bigrams, top_trigrams):
        super().__init__(cluster, docs, text_field, top_words, top_bigrams, top_trigrams)
        self.cluster = cluster

class Section(Group):

    def __init__(self, section: str, docs, text_field, top_words, top_bigrams, top_trigrams):
        super().__init__(section, docs, text_field, top_words, top_bigrams, top_trigrams)
        self.section = section

class GroupTypeError(Exception):
    pass

class Grouping(object):

    def __init__(self, docs, stoplist, text_field = 'body'):
        """
        docs: All documents to be grouped
        stoplist: list of stopwords to be omitted
        """
        self.docs = docs
        self.stoplist = stoplist
        self.text_field = text_field
        self.sections = self.__get_groups('section')
        self.clusters = self.__get_groups('label')
   
    def __get_groups(self, group_field):
        """
        group_field: Document field to be grouped by
        """
        groups_dict = self.__get_empty_groups_dict(group_field)

        # Get docs
        groups_dict_to_docs = self.__get_groups_dict_to_docs(groups_dict, group_field)
        
        # Compute top n-grams
        # Break out the nested for loop
        top_ngrams = {1: {}, 2: {}, 3: {}}
        for ngram in (1,2,3):
            top_ngrams[ngram] = self.__get_top_ngrams(top_ngrams[ngram],
                    groups_dict_to_docs,
                    ngram)

        # Build objects and return
        groups_dict_to_objects = self.__get_groups_dict_to_objects(groups_dict_to_docs,
                top_ngrams,
                group_field)

        return groups_dict_to_objects

    def __get_empty_groups_dict(self, group_field):
        """
        group_field: Document field to be grouped by
        """
        group_keys = [getattr(doc, group_field) for doc in self.docs]
        new_group_keys = group_keys
        group_keys_unique = list(set(new_group_keys))
        return {key: [] for key in group_keys_unique}
    
    def __get_groups_dict_to_docs(self, groups_dict, group_field):
        """
        group_field: Document field to be grouped by
        groups_dict: Dictionary mapping each group in this grouping to empty list
        """
        for doc in self.docs:
            group_key = getattr(doc, group_field)
            groups_dict[group_key].append(doc)
        return groups_dict

    def __get_groups_dict_to_objects(self, groups_dict, top_ngrams, group_field):
        """
        group_field: Document field to be grouped by
        groups_dict: Dictionary mapping each group to its constituent documents
        """
        field_to_type = {'section': Section, 'label': Cluster}
        group_type = field_to_type[group_field]
        
        for key in groups_dict:
            groups_dict[key] = group_type(key, groups_dict[key], 
                                         self.text_field,
                                         top_ngrams[1][key],
                                         top_ngrams[2][key],
                                         top_ngrams[3][key])
        return groups_dict

    def __get_top_ngrams(self, groups_to_ngrams, groups_dict, ngram):
        """
        groups_to_ngrams: Empty dictionary that will map each group to its top ngrams
        ngram: ngram size
        """
        bow_corpus = self.__get_bow_corpus(groups_dict, ngram)
        tfidf = models.TfidfModel(bow_corpus)
        for group in groups_dict:
            groups_to_ngrams[group] = self.__get_sorted_tfidf(tfidf, 
                    bow_corpus,
                    groups_dict[group])
        return groups_to_ngrams

    def __get_bow_corpus(self, groups_dict, ngram=1):
        """
        groups_dict: dictionary mapping groups to list of documents in the group
        ngram: Number of ngrams to build BOW corpus from
        """
        group_texts = []
        for group, group_docs in groups_dict.items():
            text = '\n\n'.join([getattr(doc, self.text_field) for doc in group_docs])
            group_texts.append(text)
        return BOWCorpus(self.stoplist, group_texts, ngram)

    def __get_sorted_tfidf(self, tfidf, bow_corpus, group_docs):
        """
        tfidf: TfIdf model to be used to get sorted tfidf for this gropu
        bow_corpus: Bag-of-words corpus containing a BOW for each group
        group_docs: List of documents in current group
        """
        group_text = '\n\n'.join([getattr(doc, self.text_field) for doc in group_docs])
        doc_bow = bow_corpus.get_bow(group_text)
        tfidf_vector = tfidf[doc_bow]
        tfidf_scores = {bow_corpus.dictionary[i] : score
                for (i, score) in tfidf_vector}
        sorted_tfidf = sorted(tfidf_scores.items(),
                key=lambda x: x[1], reverse=True)
        return sorted_tfidf        

class BOWCorpus(object):

    def __init__(self, stoplist, documents, ngram = 1):
        """
        stoplist: list of stopwords to be excluded in 1-gram case
        documents: documents with which to build BOW corpus
        ngram: size of n-grams in this corpus
        """
        self.ngram = ngram
        self.documents = []
        self.translator = str.maketrans('', '', string.punctuation + "“")
        for document in documents:
            tokens = document.lower().translate(self.translator).split()
            ngrams = self.__get_ngrams(tokens, ngram)
            self.documents.append(ngrams)

        self.dictionary = self.__get_dictionary(self.documents, stoplist)

    def __iter__(self):
        for doc in self.documents:
            yield self.dictionary.doc2bow(doc)

    def get_bow(self, document):
        """
        document: Document to be converted to BOW format in accordance with this corpus
        """
        tokens = document.lower().translate(self.translator).split()
        ngrams = self.__get_ngrams(tokens, self.ngram)
        return self.dictionary.doc2bow(ngrams)

    def __get_ngrams(self, tokens, ngram):
        """
        tokens: list of words in a document to be converted to list of ngrams
        ngram: size of ngrams
        """
        if ngram == 1:
            return tokens
        else:
            return [' '.join(tokens[i:i+ngram]) for i in range(len(tokens) - ngram)]

    def __get_dictionary(self, docs, stoplist):
        """
        docs: documents in the corpus in list of string format
        stoplist: stopwords to be used for this corpus
        """
        # Filter out punctuation
        translator = str.maketrans('', '', string.punctuation + "“")
        dictionary = corpora.Dictionary(docs)
        
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                    if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs)
                    if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids)
        dictionary.compactify()
        return dictionary
