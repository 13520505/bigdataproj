from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from .benchmarks import BenchmarkRecommender

from ..utils import max_n_sparse_indexes
import pandas as pd


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, total_epoch):
        self.epoch = 0
        self.bar = tqdm(total=total_epoch)

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        self.epoch += 1
        self.bar.update(1)


class Word2VecKNN(BenchmarkRecommender):

    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        super().__init__(clicked_items_state, params, eval_streaming_metrics)
        total_epoch = params["total_epoch"]
        window = params["window"]
        embedded_size = params["embedded_size"]
        self.iter = 0
        self.epoch = total_epoch
        self.model = Word2Vec(size=embedded_size, window=window, min_count=1,
                              workers=int(multiprocessing.cpu_count() * 0.8),
                              sg=1, iter=total_epoch, callbacks=(), compute_loss=True)

    def get_clf_suffix(self):
        return 'w2v_knn'

    def get_description(self):
        return 'Word2Vec-KNN: Most similar items sessions based on normalized cosine similarity between session ' \
               'co-occurrence learned using Word2Vec'

    def get_all_sessions_clicks(self, sessions_items, sessions_next_items):
        sessions_all_items_but_last = list([list(filter(lambda x: x != 0, session)) for session in sessions_items])
        sessions_last_item_clicked = list(
            [list(filter(lambda x: x != 0, session))[-1] for session in sessions_next_items])
        sessions_all_clicks = [previous_items + [last_item] \
                               for previous_items, last_item in
                               zip(sessions_all_items_but_last, sessions_last_item_clicked)]
        return sessions_all_clicks

    def get_sentences(self, sessions_ids, sessions_items, sessions_next_items):
        all_session_click = self.get_all_sessions_clicks(sessions_items, sessions_next_items)
        all_session_click = [list(map(str, session)) for session in all_session_click]
        return all_session_click

    def build_sentences(self, recent_click):
        df = pd.DataFrame({"item": recent_click[:,0], "session": recent_click[:,1]})
        df = df.groupby(["session"])["item"].apply(list)
        return df.tolist()

    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        self.train_model(self.get_sentences(sessions_ids, sessions_items, sessions_next_items))

    def train_model(self, sentences):
        update = False if self.iter == 0 else True
        self.model.build_vocab(sentences=sentences, update=update)
        self.model.train(sentences=sentences, total_examples=len(sentences), epochs=self.epoch)
        self.iter += 1
        # print("knn iter=", self.iter)
        # print(self.model.wv.vectors.shape[0])

    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):
        print("Predicting")
        session_predictions = np.zeros(dtype=np.int64,
                                       shape=[sessions_items.shape[0],
                                              sessions_items.shape[1],
                                              topk])

        for row_idx, session_items in enumerate(sessions_items):
            s_item = []
            for col_idx, item in enumerate(session_items):
                if item != 0:
                    item = str(item)
                    if item in self.model.wv:
                        s_item.append(item)
                    if len(s_item) == 0:
                        preds = []
                    else:
                        # Sorts items its score
                        similar_list = self.model.wv.most_similar(positive=s_item, topn=self.model.wv.vectors.shape[0])
                        preds = [int(item) for item, score in similar_list]

                    # print(preds)
                    session_predictions[row_idx, col_idx] = list(
                        self._get_top_n_valid_items(preds, topk, valid_items[row_idx, col_idx]))

        return session_predictions
