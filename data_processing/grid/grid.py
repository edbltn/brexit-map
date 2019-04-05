from gensim.utils import simple_preprocess
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer

# TODO: Switch to Document Clusters in 5 dimensions

class GridPoint(object):

    def __init__(self, grid_point, docs, average_distance):
        self.grid_point = grid_point
        self.docs = docs
        self.average_distance = average_distance
        self.text = '\n\n'.join([doc.article_text for doc in docs])

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

    def get_most_unique_terms(self, other):
        other_text = '\n'.join([o.text for o in other])
        other_tokens = simple_preprocess(other_text)
        other_set = set(other_tokens)

        tokens = simple_preprocess(self.text)

        unique_terms = list(set(tokens) - other_set)
        unique_terms_dict = {ut: self.term_frequencies[ut] 
                        for ut in unique_terms}

        return unique_terms_dict

class Grid(object):

    def __init__(self, docs, average_distance):
        self.docs = docs
        self.grid_points = self.__get_grid_points(docs,
            average_distance)
        self.sections = self.__get_sections(docs)

    def __get_grid_points(self, docs):
        grid_points = list(set([tuple(doc.grid_point) 
                                for doc in docs]))
        grid_points_dict = {gp: [] for gp in grid_points}

        for doc in docs:
            grid_points_dict[tuple(doc.grid_point)].append(doc)

        for key in grid_points_dict:
            grid_points_dict[key] = GridPoint(key, 
                grid_points_dict[key],
                average_distance[key[0], key[1]])

        return grid_points_dict

    def __get_sections(self, docs):
        return list(set([doc.section for doc in docs]))


