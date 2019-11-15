import time

import tensorflow as tf
import numpy as np


class EvalHook(tf.train.SessionRunHook):
    def __init__(self):
        super().__init__()
        self.begin_ts = None
        self.all_pred_time = None

    def begin(self):
        self.all_pred_time = []

    def before_run(self, run_context):
        self.begin_ts = time.time()

    def after_run(self, run_context, run_values):
        run_time = time.time() - self.begin_ts
        self.all_pred_time.append(run_time)

    def end(self, session):
        all_pred_time = np.array(self.all_pred_time, dtype=np.float)
        print("Min, max, avg %f - %f - %f" % (np.min(all_pred_time), np.max(all_pred_time), np.average(all_pred_time)))
        print("Total %d, took %d " % (len(all_pred_time), np.sum(all_pred_time)))
        self.all_pred_time = None
