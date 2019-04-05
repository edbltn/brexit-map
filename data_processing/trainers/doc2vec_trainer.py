from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from typing import List
import numpy as np
import pandas as pd
import re

class TaggedDocumentIterator(object):

    """Iterable of processed documents.

    Parameters
    ----------
    documents : Article list.
    """

    def __init__(self,
        documents: List[str]) -> None:

        self.documents = documents

    def __iter__(self) -> None:
        """Yields tagged processed documents.
        """
        for i, a in enumerate(self.documents):
            yield TaggedDocument(simple_preprocess(a),
                [i])

# TODO: Add new cleaning methods: NER, tokenization

class Document(object):

    """Representation of an article and all its associated data points.

    Parameters
    ----------
    index : Index of the article in the original corpus list.
    

    Attributes
    ----------
    index : See description of parameter index.

    an : str
      Article ID from DNA.

    title : str
        Title of the article from Federal Register. Contains subhead. Title and
        subhead tend to be semi-colon delimited.

    publication_date : str
        Publication date of the documentC

    _docvec : array-like, shape (vector_size,) or None
        Vector representation of the document.

    _grid_point : tuple, (x, y) or None
        Where on the SOM the document is located.

    _label : int or None
        What cluster in the SOM the document belongs to.
    """

    def __init__(self, index, row):
        self.index = index
        self.body = self.clean_body(row['body'])
        self.headline = row['headline']
        self.publication_date = row['publication_date']
        self.section = row['section']
        self.source = row['source']

    @classmethod
    def clean_body(cls, body):
        cleaned_line_breaks = re.sub(r'([^\.])\n', '\1.\n', body)
        cleaned = re.sub(r'\s\s*', ' ', cleaned_line_breaks)
        return cleaned
    
    @property
    def docvec(self):
        return self._docvec

    @docvec.setter
    def docvec(self, docvec):
        self._docvec = docvec

    @property
    def dim_one(self):
        return self._dim_one

    @dim_one.setter
    def dim_one(self, dim_one):
        self._dim_one = dim_one
    
    @property
    def dim_two(self):
        return self._dim_two

    @dim_two.setter
    def dim_two(self, dim_two):
        self._dim_two = dim_two

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def document_dict(self):
        return dict(index=self.index,
            body = self.body,
            headline = self.headline,
            doctype = self.doctype,
            publication_date = self.publication_date,
            section = self.section,
            source = self.source)

class Doc2VecTrainer(object):

    """Trains Doc2Vec model.

    Parameters
    ----------

    epochs : Number of epochs for the Doc2Vec model.

    min_count : Minimum number of times a word occurs to be included.

    vector_size : Document vector size.

    window : Window size of Doc2Vec.
    """

    def __init__(self, 
        epochs: int, 
        min_count: int, 
        vector_size: int, 
        window: int , 
        negative: int=5, 
        compute_loss: bool=False, 
        dm: int=0, 
        workers: int=5) -> None:

        self.epochs = epochs
        self.min_count = min_count
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.compute_loss = compute_loss
        self.dm = dm
        self.workers = workers
        self.corpus = None
        
    def add_corpus(self, csv_file):
        if not self.corpus:
            self.get_corpus(csv_file)
        else:
            df = pd.read_csv(csv_file)
            
            for index, row in df.iterrows():
                if not row['body'] in self.seen_articles:
                    article = Document(index, row) 
                    self.corpus.append(article)
                    self.seen_articles.add(row['body'])
                else:
                    print(row['body'][:100])

        return self

    def get_corpus(self, csv_file):
        df = pd.read_csv(csv_file)

        articles = []
        seen_articles = set()
        for index, row in df.iterrows():
            if not row['body'] in seen_articles:
                article = Document(index, row) 
                articles.append(article)
                seen_articles.add(row['body'])

        self.corpus = articles
        self.seen_articles = seen_articles

        return self
     
    def build_model(self, text_field='body'):

        """Trains Doc2Vec. Updates Doc2VecTrainer object with the model
        as a new attribute.
        """

        if not self.corpus:
            return None
        
        documents = self.__preprocess_documents(text_field)
        
        model = Doc2Vec(epochs=self.epochs,
            min_count=self.min_count,
            vector_size=self.vector_size,
            window=self.window,
            negative=self.negative,
            compute_loss=self.compute_loss,
            dm=self.dm,
            workers=self.workers)
        model.build_vocab(documents)
        model.train(documents, 
            total_examples=model.corpus_count,
            epochs=model.epochs)
        
        self.model = model

        return self

    def add_docvecs(self):

        """
        Adds document vector to each Document object in the corpus
        attribute. Updates Doc2VecTrainer object.
        """

        if not self.model:
            return None

        docvec_length = len(self.model.docvecs)
        docvecs = [self.model.docvecs[i] for i in range(docvec_length)]

        for i, docvec in enumerate(docvecs):
            self.corpus[i].docvec = docvec

        return self

    def __preprocess_documents(self, text_field) -> TaggedDocumentIterator:
        documents = self.__extract_documents(text_field)
        preprocessed_documents = TaggedDocumentIterator(documents)
        
        return preprocessed_documents

    def __extract_documents(self, text_field) -> List[str]:
        documents = [getattr(doc, text_field) for doc in self.corpus]
        return documents
