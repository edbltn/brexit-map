from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score,\
                            accuracy_score, confusion_matrix

import numpy as np

class RandomBaseline(object):

    def __init__(self, y):
        self.y = y
        self.y_pred = [randint.rvs(min(y), max(y), size=1) 
                    for lab in y]

class ExternalValidationTrainer(object):
    
    def __init__(self, an_dict, label_map,
        train_test_split=0.33, 
        baseline_statistics=None):

        self.an_dict = an_dict
        self.label_map = label_map
        self.train_test_split = train_test_split
        self.baseline_statistics = baseline_statistics

    def build_industries_dict(self, corpus_min_date):
        client = bigquery.Client()

        query_params = [
            bigquery.ScalarQueryParameter('corpus_min_date',
                'STRING', corpus_min_date)]
        job_config = bigquery.QueryJobConfig()
        job_config.query_parameters = query_params

        with open('../queries/industries.sql', 'r') as f:
            query = f.read()

        query_job = client.query(
            query,
            location='US',
            job_config=job_config)

        data_dict = {row.an: self.__unpack_row(row) 
                    for row in query_job}
        self.data_dict = data_dict

        return self

    @property
    def data(self):
        if not self.data_dict:
            return None

        X = []
        y = []
        for key in self.data_dict:
            X.append(data_dict[key]['docvec'])
            y.append(data_dict[key]['label'])

        return X, y

    def __unpack_row(self, row):
        unpacked_row = dict(label=self.label_map[row.label],
                    docvec=self.an_dict[row.an].get('docvec', None))
        return unpacked_row