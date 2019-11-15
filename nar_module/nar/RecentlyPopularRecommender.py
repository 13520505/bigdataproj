import numpy as np
from collections import Counter

from nar_module.nar.nar_model import get_list_id


class RecentlyPopularRecommender():
    def __init__(self):
        pass

    def get_recent_popular_item_ids(self):
        # get item recent buffer from google service
        # recent_items_buffer = self.clicked_items_state.get_recent_clicks_buffer()
        recent_items_buffer = get_list_id()
        #recent_items_buffer_nonzero = recent_items_buffer[np.nonzero(recent_items_buffer)]
        #Dealing with first batch, when there is no item in the buffer yet
        #if len(recent_items_buffer_nonzero) == 0:
        #    recent_items_buffer_nonzero = [0]
        item_counter = Counter(recent_items_buffer)
        popular_item_ids, popular_items_count = zip(*item_counter.most_common())
        #print(len(recent_items_buffer))
        return popular_item_ids

    def get_top_k_items_pop(self, popular_item_ids):
        return popular_item_ids[:100]

    def _get_top_n_valid_items(self, items, topk, valid_items):
        count = 0
        for item in items:
            if count == topk:
                break
            if (item in valid_items) or (valid_items is None):
                count += 1
                yield item

    def predict(self, users_ids, sessions_items, topk=100, valid_items=None):
        popular_item_ids = self.get_recent_popular_item_ids()

        session_predictions = np.zeros(dtype=np.int64,
                                       shape=[sessions_items.shape[0],
                                              sessions_items.shape[1],
                                              topk])

        for row_idx, session_items in enumerate(sessions_items):
            for col_idx, item in enumerate(session_items):
                if item != 0:
                    session_predictions[row_idx, col_idx] = list(
                        self._get_top_n_valid_items(popular_item_ids, topk, valid_items[row_idx, col_idx]))

        return session_predictions
