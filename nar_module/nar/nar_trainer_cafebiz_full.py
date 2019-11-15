from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Disabling GPU for local execution
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from time import time
import tensorflow as tf
import json
import os
import numpy as np
import tempfile
import logging
import shutil

from acr_module.acr.acr_module_service import load_json_config
from pick_singleton.pick_singleton import ACR_Pickle_Singleton, NAR_Pickle_Singleton
from redis_connector.RedisClient import Singleton
from .utils import resolve_files, chunks, log_elapsed_time, append_lines_to_text_file, min_max_scale, get_current_time
from .datasets import prepare_dataset_iterator, prepare_dataset_generator_predict, parse_sequence_example_predict
from .nar_model import ClickedItemsState, ItemsStateUpdaterHook, NARModuleModel,get_list_id, get_tf_dtype
from .benchmarks import RecentlyPopularRecommender, ContentBasedRecommender, ItemCooccurrenceRecommender, ItemKNNRecommender, SessionBasedKNNRecommender, SequentialRulesRecommender

from .nar_utils import load_acr_module_resources, load_nar_module_preprocessing_resources, save_eval_benchmark_metrics_csv, \
        upload_model_output_to_gcs, dowload_model_output_from_gcs

import glob        

tf.logging.set_verbosity(tf.logging.INFO)

#Making results reproduceable
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)

#Model params
tf.flags.DEFINE_integer('batch_size', default=64, help='Batch size')
tf.flags.DEFINE_integer('truncate_session_length', default=20, help='Truncate long sessions to this max. size')
tf.flags.DEFINE_float('learning_rate', default=1e-3, help='Lerning Rate')
tf.flags.DEFINE_float('dropout_keep_prob', default=1.0, help='Dropout (keep prob.)')
tf.flags.DEFINE_float('reg_l2', default=0.0001, help='L2 regularization')
tf.flags.DEFINE_float('softmax_temperature', default=0.2, help='Initial value for temperature for softmax')
tf.flags.DEFINE_float('recent_clicks_buffer_hours', default=1.0, help='Number of hours that will be kept in the recent clicks buffer (limited by recent_clicks_buffer_max_size)')
tf.flags.DEFINE_integer('recent_clicks_buffer_max_size', default=6000, help='Maximum size of recent clicks buffer')
tf.flags.DEFINE_integer('recent_clicks_for_normalization', default=2500, help='Number of recent clicks to normalize recency and populary  novelty) dynamic features')
tf.flags.DEFINE_integer('eval_metrics_top_n', default=100, help='Eval. metrics Top N')
tf.flags.DEFINE_integer('CAR_embedding_size', default=1024, help='CAR submodule embedding size')
tf.flags.DEFINE_integer('rnn_units', default=10, help='Number of units of RNN cell')
tf.flags.DEFINE_integer('rnn_num_layers', default=1, help='Number of of RNN layers')
tf.flags.DEFINE_integer('train_total_negative_samples', default=7, help='Total negative samples for training')
tf.flags.DEFINE_integer('train_negative_samples_from_buffer', default=10, help='Training Negative samples from recent clicks buffer')
tf.flags.DEFINE_integer('eval_total_negative_samples', default=600, help='Total negative samples for evaluation')
tf.flags.DEFINE_integer('eval_negative_samples_from_buffer', default=5000, help='Eval. Negative samples from recent clicks buffer')
tf.flags.DEFINE_bool('save_histograms', default=False, help='Save histograms to view on Tensorboard (make job slower)')
tf.flags.DEFINE_bool('disable_eval_benchmarks', default=True, help='Disable eval benchmarks')
tf.flags.DEFINE_bool('eval_metrics_by_session_position', default=False, help='Computes eval metrics at each position within session (e.g. 1st click, 2nd click)')
tf.flags.DEFINE_float('novelty_reg_factor', default=0.0, help='Popularity Regularization Loss (e.g. 0.1, 0.2, 0.3)')
tf.flags.DEFINE_float('diversity_reg_factor', default=0.0, help='Diversity (similarity) Regularization Loss (e.g. 0.1, 0.2, 0.3)')
tf.flags.DEFINE_float('eval_negative_sample_relevance', default=0.1, help='Relevance of negative samples within top-n recommended items for evaluation (relevance of positive sample is always 1.0)')
tf.flags.DEFINE_string('current_time', default=get_current_time(), help='get current running time for save top-n recommendation')
tf.flags.DEFINE_bool('save_eval_per_sessions', default=False, help='Save last batch evaluation for each session hours trainning')

tf.flags.DEFINE_list('enabled_clicks_input_features_groups',
                    default='time,location', help='Groups of input contextual features for user clicks, separated by comma. Valid values: ALL,NONE,time,device,location,referrer')
tf.flags.DEFINE_list('enabled_articles_input_features_groups',
                    default='category', help='Groups of input metadata features for articles, separated by comma. Valid values: ALL,NONE,category,author')
tf.flags.DEFINE_list('enabled_internal_features',
                    default='recency,novelty,article_content_embeddings', help='Internal features. Valid values: ALL,NONE,recency,novelty,article_content_embeddings,item_clicked_embeddings')



#Control params
#tf.flags.DEFINE_string('data_dir', default_value='./tmp',
#                    help='Directory where the dataset is located')
tf.flags.DEFINE_string('train_set_path_regex',
                    default='/train*.tfrecord', help='Train set regex')
tf.flags.DEFINE_string('acr_module_resources_path',
                    default='/pickles', help='ACR module resources path')
tf.flags.DEFINE_string('nar_module_preprocessing_resources_path',
                    default='/pickles', help='NAR module preprocessing resources path')
tf.flags.DEFINE_string('model_dir', default='./tmp',
                    help='Directory where save model checkpoints')
tf.flags.DEFINE_string('warmup_model_dir', default=None,
                    help='Directory where model checkpoints of a previous job where output, to warm start this network training')

tf.flags.DEFINE_integer('train_files_from', default=0, help='Train model starting from file N')
tf.flags.DEFINE_integer('train_files_up_to', default=100, help='Train model up to file N')
tf.flags.DEFINE_integer('training_hours_for_each_eval', default=5, help='Train model for N hours before evaluation of the next hour')
tf.flags.DEFINE_integer('save_results_each_n_evals', default=5, help='Saves to disk and uploads to GCS (ML Engine) the incremental evaluation results each N evaluations')
tf.flags.DEFINE_bool('save_eval_sessions_negative_samples', default=False, help='Save negative samples of each session during evaluation')
tf.flags.DEFINE_bool('save_eval_sessions_recommendations', default=False, help='Save CHAMELEON recommendations log during evaluation')
tf.flags.DEFINE_bool('use_local_cache_model_dir', default=False, help='Persists checkpoints and events in a local temp file, copying to GCS in the end of the process (useful for ML Engine jobs, because saving and loading checkpoints slows training job)')
#Default param used by ML Engine to validate whether the path exists
tf.flags.DEFINE_string('job-dir', default='./tmp', help='Job dir to save staging files')

tf.flags.DEFINE_bool('prediction_only', default=False, help='Experimental prediction only mode')


FLAGS = tf.flags.FLAGS
#params_dict = tf.app.flags.FLAGS.flag_values_dict()
#tf.logging.info('PARAMS: {}'.format(json.dumps(params_dict)))

ALL_FEATURES = 'ALL'

def get_articles_features_config(acr_label_encoders):
    articles_features_config = {
        #Required fields
        'article_id': {'type': 'categorical', 'dtype': 'int'},
        'created_at_ts': {'type': 'numerical', 'dtype': 'int'},
        #Additional metadata fields
        'category0': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 41},
        # 'category1': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 128},
        # 'author': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 112},
    }

    feature_groups = {
        'category': ['category0'],
        # 'category': ['category0', 'category1'],
        # 'author': ['author'],
    }

    #Disabling optional features when required
    if FLAGS.enabled_articles_input_features_groups != [ALL_FEATURES]:   
        for feature_group in feature_groups:
            if feature_group not in FLAGS.enabled_articles_input_features_groups:
                for feature in feature_groups[feature_group]:
                    del articles_features_config[feature]

    #Adding cardinality to categorical features
    for feature_name in articles_features_config:
        if feature_name in acr_label_encoders and articles_features_config[feature_name]['type'] == 'categorical':
            articles_features_config[feature_name]['cardinality'] = len(acr_label_encoders[feature_name])

    # tf.logging.info('Article Features: {}'.format(articles_features_config))
    return articles_features_config


def process_articles_metadata(articles_metadata_df, articles_features_config):
    articles_metadata = {}
    for feature_name in articles_features_config:
        articles_metadata[feature_name] = articles_metadata_df[feature_name].values
        #Appending a row in the first position to correspond to the <PAD> article #
        # (so that it correspond to content_article_embeddings_matrix.shape[0])
        articles_metadata[feature_name] = np.hstack([[0], articles_metadata[feature_name]])
    return articles_metadata

def get_session_features_config(nar_label_encoders_dict):
    session_features_config = {
        'single_features': {
            #Control features
            'user_id': {'type': 'categorical', 'dtype': 'bytes'},
            'session_id': {'type': 'numerical', 'dtype': 'int'},
            'session_size': {'type': 'numerical', 'dtype': 'int'},
            'session_start': {'type': 'numerical', 'dtype': 'int'},            
        },
        'sequence_features': {
            #Required sequence features
            'event_timestamp': {'type': 'numerical', 'dtype': 'int'},
            'item_clicked': {'type': 'categorical', 'dtype': 'int'},#, 'cardinality': 72933},

            #Location        
            'city': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 1022}, 
            # 'region': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 237},
            # 'country': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 70},
            
            #Device
            # 'device': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 5},
            'os': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 10}, 
            
            #Time
            'local_hour_sin': {'type': 'numerical', 'dtype': 'float'},
            'local_hour_cos': {'type': 'numerical', 'dtype': 'float'},
            'weekday': {'type': 'numerical', 'dtype': 'float'},

            #Referrer type
            # 'referrer_class': {'type': 'categorical', 'dtype': 'int'}, #'cardinality': 7}}}
        }
    }


    feature_groups = {
        'time': ['local_hour_sin', 'local_hour_cos', 'weekday'],
        'device': ['os'],
        'location': ['city'],
        'referrer': []


        # 'device': ['device', 'os'],
        # 'location': ['country', 'region', 'city'],
        # 'referrer': ['referrer_class']
    }


    #Disabling optional features when required
    if FLAGS.enabled_clicks_input_features_groups != [ALL_FEATURES]:   
        for feature_group in feature_groups:
            if feature_group not in FLAGS.enabled_clicks_input_features_groups:
                for feature in feature_groups[feature_group]:
                    del session_features_config['sequence_features'][feature]


    #Adding cardinality to categorical features
    for feature_groups_key in session_features_config:
        features_group_config = session_features_config[feature_groups_key]
        for feature_name in features_group_config:
            if feature_name in nar_label_encoders_dict and features_group_config[feature_name]['type'] == 'categorical':
                features_group_config[feature_name]['cardinality'] = len(nar_label_encoders_dict[feature_name])

    # tf.logging.info('Session Features: {}'.format(session_features_config))

    return session_features_config
    
def get_internal_enabled_features_config():
    VALID_INTERNAL_FEATURES = ['recency','novelty','article_content_embeddings','item_clicked_embeddings']
    internal_features_config = {}
    enabled_features = []
    if FLAGS.enabled_internal_features == [ALL_FEATURES]:
        enabled_features = set(VALID_INTERNAL_FEATURES)
    else:
        enabled_features = set(FLAGS.enabled_internal_features).intersection(set(VALID_INTERNAL_FEATURES))
    for feature in VALID_INTERNAL_FEATURES:
        internal_features_config[feature] = (feature in enabled_features)
    tf.logging.info('Enabled internal features: {}'.format(enabled_features))
    return internal_features_config


def nar_module_model_fn(features, labels, mode, params):    
    if mode == tf.estimator.ModeKeys.TRAIN:
        negative_samples = params['train_total_negative_samples']
        negative_sample_from_buffer = params['train_negative_samples_from_buffer']
    elif mode == tf.estimator.ModeKeys.EVAL:
        negative_samples = params['eval_total_negative_samples']
        negative_sample_from_buffer = params['eval_negative_samples_from_buffer']
    elif mode == tf.estimator.ModeKeys.PREDICT:
        negative_samples = params['eval_total_negative_samples']
        negative_sample_from_buffer = params['eval_negative_samples_from_buffer']

    
    dropout_keep_prob = params['dropout_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0
    
    internal_features_config = get_internal_enabled_features_config()
    
    eval_metrics_top_n = params['eval_metrics_top_n']
    
    model = NARModuleModel(mode, features, labels,
              session_features_config=params['session_features_config'],
              articles_features_config=params['articles_features_config'],
              batch_size=params['batch_size'], 
              lr=params['lr'],
              keep_prob=dropout_keep_prob,
              negative_samples=negative_samples,
              negative_sample_from_buffer=negative_sample_from_buffer,
              reg_weight_decay=params['reg_weight_decay'], 
              softmax_temperature=params['softmax_temperature'], 
              articles_metadata=params['articles_metadata'],
              content_article_embeddings_matrix=params['content_article_embeddings_matrix'],
              recent_clicks_buffer_hours=params['recent_clicks_buffer_hours'],
              recent_clicks_buffer_max_size=params['recent_clicks_buffer_max_size'],
              recent_clicks_for_normalization=params['recent_clicks_for_normalization'],
              CAR_embedding_size=params['CAR_embedding_size'],
              rnn_units=params['rnn_units'],
              # metrics_top_n=100,
              metrics_top_n=eval_metrics_top_n,
              plot_histograms=params['save_histograms'],
              novelty_reg_factor=params['novelty_reg_factor'],
              diversity_reg_factor=params['diversity_reg_factor'], 
              internal_features_config=internal_features_config
             )
    
    #Using these variables as global so that they persist across different train and eval
    global clicked_items_state, eval_sessions_metrics_log, sessions_negative_items_log

    eval_benchmark_classifiers = []
    if not FLAGS.disable_eval_benchmarks:
        eval_benchmark_classifiers=[
            #{'recommender': Word2VecKNN, 'params': {"total_epoch": 100, "window": 5, "embedded_size":300}},
                                    {'recommender': RecentlyPopularRecommender, 'params': {}},
                                    {'recommender': ItemCooccurrenceRecommender, 'params': {}},
                                    {'recommender': ItemKNNRecommender, 
                                          'params': {'reg_lambda': 20,  #Regularization. Discounts the similarity of rare items (incidental co-occurrences). 
                                                     'alpha': 0.5 #Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
                                                     }},
                                    {'recommender': SessionBasedKNNRecommender, 
                                          'params': {'sessions_buffer_size': 3000, #Buffer size of last processed sessions
                                                     'candidate_sessions_sample_size': 2000, #200, #Number of candidate near sessions to sample  
                                                     'sampling_strategy': 'recent', #(recent,random)
                                                     'nearest_neighbor_session_for_scoring': 500, #50 #Nearest neighbors to compute item scores    
                                                     'similarity': 'cosine', #(jaccard, cosine)
                                                     'first_session_clicks_decay': 'div' #Decays weight of first user clicks in active session when finding neighbor sessions (same, div, linear, log, quadradic)
                                                     }},
                                    {'recommender': ContentBasedRecommender, 
                                          'params': {'articles_metadata': params['articles_metadata'],
                                                     'content_article_embeddings_matrix': params['content_article_embeddings_matrix']}},
                                    {'recommender': SequentialRulesRecommender,
                                          'params': {'max_clicks_dist': 10, #Max number of clicks to walk back in the session from the currently viewed item. (Default value: 10) 
                                                     'dist_between_clicks_decay': 'div' #Decay function for distance between two items clicks within a session (linear, same, div, log, qudratic). (Default value: div) 
                                                     }}
                                   ]
                                                              
    hooks = [ItemsStateUpdaterHook(mode, model, 
                                   eval_metrics_top_n=eval_metrics_top_n,
                                   clicked_items_state=clicked_items_state, 
                                   eval_sessions_metrics_log=eval_sessions_metrics_log,
                                   sessions_negative_items_log=sessions_negative_items_log,
                                   sessions_chameleon_recommendations_log=sessions_chameleon_recommendations_log,
                                   content_article_embeddings_matrix=params['content_article_embeddings_matrix'],
                                   articles_metadata=params['articles_metadata'],
                                   eval_negative_sample_relevance=params['eval_negative_sample_relevance'],
                                   global_eval_hour_id=global_eval_hour_id,
                                   eval_benchmark_classifiers=eval_benchmark_classifiers,
                                   eval_metrics_by_session_position=params['eval_metrics_by_session_position']
                                   )] 
    
    if mode == tf.estimator.ModeKeys.TRAIN:        
        return tf.estimator.EstimatorSpec(mode, loss=model.total_loss, train_op=model.train,
                                      training_chief_hooks=hooks)

    elif mode == tf.estimator.ModeKeys.EVAL:  

        eval_metrics = {#'hitrate_at_1': (model.next_item_accuracy_at_1, model.next_item_accuracy_at_1_update_op),
                        'hitrate_at_n': (model.recall_at_n, model.recall_at_n_update_op),
                        'mrr_at_n': (model.mrr, model.mrr_update_op),   
                        #'ndcg_at_n': (model.ndcg_at_n_mean, model.ndcg_at_n_mean_update_op),                 
                       }
                        
        return tf.estimator.EstimatorSpec(mode, loss=model.total_loss, eval_metric_ops=eval_metrics,
                                      evaluation_hooks=hooks) 
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=model.predictions,prediction_hooks=hooks) 


def build_estimator(model_dir,
    content_article_embeddings_matrix, 
    articles_metadata, articles_features_config,
    session_features_config):
    """Build an estimator appropriate for the given model type."""

    #Disabling GPU (memory issues on local machine)
    #config_proto = tf.ConfigProto(device_count={'GPU': 0})    
    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        keep_checkpoint_max=1, 
                                        save_checkpoints_secs=1200, 
                                        save_summary_steps=100,
                                        log_step_count_steps=100,
                                        #session_config=config_proto
                                        )

    estimator = tf.estimator.Estimator(
        config=run_config,
        model_dir=model_dir,
        model_fn=nar_module_model_fn,    
        params={
            'batch_size': FLAGS.batch_size,
            'lr': FLAGS.learning_rate,
            'dropout_keep_prob': FLAGS.dropout_keep_prob,
            'reg_weight_decay': FLAGS.reg_l2,
            'recent_clicks_buffer_hours': FLAGS.recent_clicks_buffer_hours,
            'recent_clicks_buffer_max_size': FLAGS.recent_clicks_buffer_max_size,
            'recent_clicks_for_normalization': FLAGS.recent_clicks_for_normalization,
            'eval_metrics_top_n': FLAGS.eval_metrics_top_n,
            'CAR_embedding_size': FLAGS.CAR_embedding_size,
            'rnn_units': FLAGS.rnn_units,
            'train_total_negative_samples': FLAGS.train_total_negative_samples,
            'train_negative_samples_from_buffer': FLAGS.train_negative_samples_from_buffer,
            'eval_total_negative_samples': FLAGS.eval_total_negative_samples,
            'eval_negative_samples_from_buffer': FLAGS.eval_negative_samples_from_buffer,
            'softmax_temperature': FLAGS.softmax_temperature,
            'save_histograms': FLAGS.save_histograms,
            'eval_metrics_by_session_position': FLAGS.eval_metrics_by_session_position,
            'novelty_reg_factor': FLAGS.novelty_reg_factor,
            'diversity_reg_factor': FLAGS.diversity_reg_factor,
            'eval_negative_sample_relevance': FLAGS.eval_negative_sample_relevance,

            #From pre-processing
            'session_features_config': session_features_config,
            'articles_features_config': articles_features_config,
            'articles_metadata': articles_metadata,            
            #From ACR module
            'content_article_embeddings_matrix': content_article_embeddings_matrix
        })

    return estimator


#Saving the negative samples used to evaluate each sessions, so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
def save_sessions_negative_items(model_output_dir, sessions_negative_items_list, output_file='eval_sessions_negative_samples.json'):
    append_lines_to_text_file(os.path.join(model_output_dir, output_file), 
                                           map(lambda x: json.dumps({'session_id': x['session_id'],
                                                                     'negative_items': x['negative_items']}), 
                                               sessions_negative_items_list))


def save_sessions_chameleon_recommendations_log(model_output_dir, sessions_chameleon_recommendations_log_list, 
                                                eval_hour_id, output_file='eval_chameleon_recommendations_log.json'):
    append_lines_to_text_file(os.path.join(model_output_dir, output_file), 
                                           map(lambda x: json.dumps({'eval_hour_id': eval_hour_id,
                                                                     'session_id': x['session_id'],
                                                                     'next_click_labels': x['next_click_labels'],
                                                                     'predicted_item_ids': x['predicted_item_ids'],
                                                                     'predicted_item_probs': x['predicted_item_probs'],
                                                                     'predicted_item_norm_pop': x['predicted_item_norm_pop']
                                                                     }), 
                                               sessions_chameleon_recommendations_log_list))


#Global vars updated by the Estimator Hook
clicked_items_state = None
eval_sessions_metrics_log = [] 
sessions_negative_items_log = [] if FLAGS.save_eval_sessions_negative_samples else None
sessions_chameleon_recommendations_log = [] if FLAGS.save_eval_sessions_recommendations else None
global_eval_hour_id = 0

import threading
from queue import Queue
from threading import Thread

#Export model for multithread predict
def export_model(estimator, export_dir):
    def _serving_input_receiver_fn():
        nar_label_encoders = \
            NAR_Pickle_Singleton.getInstance()
        session_features_config = get_session_features_config(nar_label_encoders)
        serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None, 
                                           name='input_example_tensor')
        # key (e.g. 'examples') should be same with the inputKey when you 
        # buid the request for prediction
        receiver_tensors = {'examples': serialized_tf_example}
        # features = tf.parse_example(serialized_tf_example,session_features_config)
        features = parse_sequence_example_predict(serialized_tf_example, session_features_config)[0]
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(features)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    estimator.export_saved_model(export_dir,_serving_input_receiver_fn)

# Prediction using session
def get_session(exported_path):
    sess = tf.Session()
    pickle = ACR_Pickle_Singleton.getInstance()
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
    return sess

class FastClassifierThreaded():

    def __init__(self, estimator,
                 threaded=True,
                 verbose=False):
        """
        Parameters
        ----------
        model_path: str
            Location from which to load the model.
        threaded: Boolean [True]
            Whether to use multi-threaded execution for inference.
            If False, the model will use a new generator for each sample that is passed to it, and reload the entire
            model each time.
        """

        # super(FlowerClassifierThreaded, self).__init__(model_path=model_path,
        #                                                verbose=verbose)
        self.estimator = estimator
        self.verbose = verbose
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        # Kill thread when true
        self.killed = False
        self.threaded = threaded

        if self.threaded:
            # We set the generator thread as daemon
            # (see https://docs.python.org/3/library/threading.html#threading.Thread.daemon)
            # This means that when all other threads are dead,
            # this thread will not prevent the Python program from exiting
            # print("ACTIVE NUM THREAD: %d" % (threading.active_count()))
            self.prediction_thread = Thread(target=self.predict_from_queue, daemon=False, args=(lambda: self.killed, ))
            # print("THREAD NAME: "+self.prediction_thread.getName())
            self.prediction_thread.start()
        else:
            self.predictions = self.estimator.predict(input_fn=lambda: self.queued_predict_input_fn(lambda: self.killed))
        # print("FastClassifierThread init success.")
    
    def kill_thread(self):
        self.killed = True

    def generate_from_queue(self,stop):
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.
        """

        while True:
            try:
                # print("KILLED: %r" % self.killed)
                # print("STOPPED: %r" % stop())
                # if self.killed:
                #     print("REMOVE THREAD")
                #     break
                if stop():
                    # print("STOP THREAD")
                    return
                    yield
                if self.verbose:
                    print('Yielding from input queue')
                yield self.input_queue.get()
            except Exception as e:
                print("Err queue")
                print(e)
                yield ""
            # finally:
            #     print("Error queue")
            #     self.input_queue.task_done()
    def predict_from_queue(self,stop):
        # print("THREAD ID: %d" %threading.current_thread().ident)
        """ Adds a prediction from the model to the output_queue.
        This lives within our 'prediction thread'.
        Note: estimators accept generators as inputs and return generators as output. Here, we are
        iterating through the output generator, which will be populated in lock-step with the input
        generator.
        """
        while not stop():
            try:
                for i in self.estimator.predict(input_fn=lambda: self.queued_predict_input_fn(stop)):
                    if self.verbose:
                        print('Putting in output queue')
                    self.output_queue.put(i)
                    print(stop())
                    print(stop)
                    if stop():
                        print("STOP THREAD OUTPUT")
                        # raise StopIteration
                print("OUT FOR LOOP")
                break
            except Exception as ex:
                print("Exception predict_from_queue")
                print(ex)
                raise

    def predict(self, features):
        """
        Overwrites .predict in FlowerClassifierBasic.
        Calls either the vanilla or multi-threaded prediction methods based upon self.threaded.
        Parameters
        ----------
        features: dict
            dict of input features, containing keys 'SepalLength'
                                                    'SepalWidth'
                                                    'PetalLength'
                                                    'PetalWidth'
        Returns
        -------
        predictions: dict
            Dictionary containing   'probs'
                                    'outputs'
                                    'predicted_class'
        """
        try:
            # Get predictions dictionary
            # print("Into FastClassifierThreaded.predict() ")
            # print(self.input_queue.qsize())
            # print(self.output_queue.qsize())
            if self.threaded:
                # features = dict(features)
                self.input_queue.put(features)
                predictions = self.output_queue.get()  # The latest predictions generator
            else:
                predictions = next(self.predictions)
                # predictions = self.estimator.predict(input_fn=lambda: self.predict_input_fn(features))

            # print("Prediction in FastClassifierThreaded.Predict() ")
            # print(predictions)
            # TODO: list vs. generator vs. dict handling
            return predictions
        except Exception as ex:
            print("Exception predict func")
            print(ex)
            raise

    def queued_predict_input_fn(self,stop):
        """
        Queued version of the `predict_input_fn` in FlowerClassifier.
        Instead of yielding a dataset from data as a parameter, we construct a Dataset from a generator,
        which yields from the input queue.
        """

        if self.verbose:
            print("QUEUED INPUT FUNCTION CALLED")

        # Fetch the inputs from the input queue
        def _inner_input_fn():
            nar_label_encoders = \
                NAR_Pickle_Singleton.getInstance()
            session_features_config = get_session_features_config(nar_label_encoders)
            return prepare_dataset_generator_predict(lambda: self.generate_from_queue(stop), session_features_config, batch_size=1, 
            truncate_session_length=FLAGS.truncate_session_length, predict_only=True)
        return _inner_input_fn()
# tungtv Class Nar Model SingleTon For Predict
class NAR_Model_Predict(object,metaclass=Singleton):
    __instance = None
    lock = threading.Lock()
    def __init__(self):
        print("NAR Model Predict init ... ")
        pickle = ACR_Pickle_Singleton.getInstance()
        model_output_dir =pickle.model_nar_dir

        acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = pickle.acr_label_encoders, pickle.articles_metadata_df, pickle.content_article_embeddings_matrix
        self.content_article_embeddings_matrix = min_max_scale(content_article_embeddings_matrix, min_max_range=(-0.1, 0.1))
        articles_features_config = get_articles_features_config(acr_label_encoders)
        articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)

        nar_label_encoders = \
            NAR_Pickle_Singleton.getInstance()
        session_features_config = get_session_features_config(nar_label_encoders)
        global clicked_items_state
        clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_hours,
                                FLAGS.recent_clicks_buffer_max_size,
                                FLAGS.recent_clicks_for_normalization,
                                    self.content_article_embeddings_matrix.shape[0])
        estimator = build_estimator(model_output_dir,
                            self.content_article_embeddings_matrix, articles_metadata, articles_features_config,
                            session_features_config)
        # export_dir = model_output_dir+"exported/"
        model_predict_path = model_output_dir + "exported/"
        from os import path
        if path.exists(model_predict_path):
            pass
        else:
            os.makedirs(model_predict_path)
        #  create dir for each works
        import random
        from os import path
        while True:
            num = random.randint(1, 1000)
            export_dir = model_predict_path + str(num) + '/'
            if not path.exists(export_dir):
                break
        self.export_dir = export_dir
        #Remove old files
        # shutil.rmtree(export_dir)
        export_model(estimator, self.export_dir)
        
        # Get in subfolder of exported-model create
        model_paths = glob.glob(export_dir+"*")
        lastest = sorted(model_paths)[-1]
        self.session = get_session(lastest)
        #Load tensor for predict
        self.load_tensor_from_session(self.session)
        self.predict_fn = self.predict_from_expoted_model

        #Load internal features
        self.load_internal_features()
        # print("NAR Model Predict build_estimator done")
        #Add FastPred
        # self.model = FastClassifierThreaded(estimator, verbose=True)
        # print("Model = FastClassifierThread Done")
        NAR_Model_Predict.__instance = self
    def load_tensor_from_session(self,sess):
        #Define output tensor
        self.topk = sess.graph.get_tensor_by_name("main/recommendations_ranking/predicted_items/Reshape:0")
        self.user_id = sess.graph.get_tensor_by_name("cond/Merge:0")
        self.item_clicked = sess.graph.get_tensor_by_name("ExpandDims_5:0")
        self.fetches = {"top_k_predictions":self.topk,"user_id":self.user_id,"item_clicked":self.item_clicked}

        #Define features
        self.examples = sess.graph.get_tensor_by_name("input_example_tensor:0")
        self.recent_item_buffer = sess.graph.get_tensor_by_name("articles_status/pop_recent_items_buffer:0")
        self.articles_recent_pop_norm = sess.graph.get_tensor_by_name("articles_status/articles_recent_pop_norm:0")
        self.content_article_embeddings_matrix_ts = sess.graph.get_tensor_by_name("article_content_embeddings/content_article_embeddings_matrix:0")
        # print("shape self.content_article_embeddings_matrix_ts")
        # print(self.content_article_embeddings_matrix_ts.get_shape())
        # articles_metadata0 = sess.graph.get_tensor_by_name("article_content_embeddings/articles_metadata_0:0")
        self.articles_metadata1 = sess.graph.get_tensor_by_name("article_content_embeddings/articles_metadata_1:0")
        self.articles_metadata2 = sess.graph.get_tensor_by_name("article_content_embeddings/articles_metadata_2:0")
    
    def load_internal_features(self):
        pickle = ACR_Pickle_Singleton.getInstance()
        self.articles_features_config = get_articles_features_config(pickle.acr_label_encoders)
        self.articles_metadata = process_articles_metadata(pickle.articles_metadata_df, self.articles_features_config)
    @staticmethod
    def getInstance():
        if NAR_Model_Predict.__instance == None:
            print("NAR singleton is none")
            NAR_Model_Predict()
        return NAR_Model_Predict.__instance
    
    def predict_from_expoted_model(self, sess, news_id, guid):
        # Get session and parse features from newsid, guid
        parsed_example = parse_feature_to_string(news_id,guid) 
        
        # Offline local test       
        # recent_clicks  = list(np.random.randint(low=1, high=2800, size=20000))
        # recency = clicked_items_state.get_recent_pop_norm_for_predict(recent_clicks)

        # Online gg services
        # recent_clicks = get_list_id()
        # encoded_list_id = []
        # count = 0
        #convert id to encoded id
        pickle = ACR_Pickle_Singleton.getInstance()


        # for id in recent_clicks:
        #     if(int(id) in pickle.acr_label_encoders['article_id']):
        #         encoded_list_id.append(pickle.get_article_id_encoded(int(id)))
        #     else:
        #         count = count +1

        encoded_list_id = pickle.encoded_list_id
        recency = clicked_items_state.get_recent_pop_norm_for_predict(encoded_list_id)

        feed_dict={self.examples:parsed_example,self.recent_item_buffer:recency[1],
                    self.articles_recent_pop_norm:recency[0],
                    self.content_article_embeddings_matrix_ts:self.content_article_embeddings_matrix,
                    # articles_metadata0:articles_metadata['article_id'],
                    self.articles_metadata1:self.articles_metadata['created_at_ts'],
                    self.articles_metadata2:self.articles_metadata['category0']}

        
        predictions = sess.run(self.fetches, feed_dict)
        return predictions
        # predictor = tf.contrib.predictor.from_saved_model(exported_path)
        # input_tensor=tf.get_default_graph().get_tensor_by_name("input_example_tensor:0")
        # output_dict= predictor({"examples":parsed_example})
        # return output_dict

    def getUpdateInstance(self):
        print("===>NAR MODEL PREDICT UPDATE")
        pickle = ACR_Pickle_Singleton.getInstance()
        model_output_dir = pickle.model_nar_dir
        # print("Model dir: {}".format(model_output_dir))

        acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = pickle.acr_label_encoders, pickle.articles_metadata_df, pickle.content_article_embeddings_matrix
        self.content_article_embeddings_matrix = min_max_scale(content_article_embeddings_matrix, min_max_range=(-0.1, 0.1))
        articles_features_config = get_articles_features_config(acr_label_encoders)
        articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)

        # print("matrix: ")
        # print(content_article_embeddings_matrix.shape[0])
        # print("Shape self")
        # print(self.content_article_embeddings_matrix.shape[0])
        nar_label_encoders = \
            NAR_Pickle_Singleton.getInstance()
        session_features_config = get_session_features_config(nar_label_encoders)

        global clicked_items_state
        clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_hours,
                                                FLAGS.recent_clicks_buffer_max_size,
                                                FLAGS.recent_clicks_for_normalization,
                                                self.content_article_embeddings_matrix.shape[0])
        # print("NUM ITEMS CLICKED STATE: ")
        # print(clicked_items_state.num_items)
        estimator = build_estimator(model_output_dir,
                            self.content_article_embeddings_matrix, articles_metadata, articles_features_config,
                            session_features_config)
        # print("Into NAR Model Update")
        
        old_session = self.session

        #Remove old files
        # shutil.rmtree(export_dir)
        for root, dirs, files in os.walk(self.export_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        export_model(estimator, self.export_dir)
        
        # Get in subfolder of exported-model create
        model_paths = glob.glob(self.export_dir+"*")
        lastest = sorted(model_paths)[-1]
        session = get_session(lastest)
        self.session = session

        # Load tensor for predict
        self.load_tensor_from_session(self.session)
        self.predict_fn = self.predict_from_expoted_model

        #Reload internal features for new session
        self.load_internal_features()
        #self.content_article_embeddings_matrix = content_article_embeddings_matrix
        #Close old session
        old_session.close()
        print("UPDATE NAR MODEL PREDICT DONE")

        # old_model = self.model
        # #Add FastPred
        # model = FastClassifierThreaded(estimator, verbose=True)
        # # print("Start predict first time")
        # self.model = model
        # # print(model.prediction_thread)
        # NAR_Model_Predict.__instance = self
        # # Example predict for first time to load graph (avoid lazy load))
        # predict("20190918203115156","reloadUser:0")
        # # print("ReLoad NAR MODEL Predict done")
        
        # # Kill old thread
        # old_model.kill_thread()
        # # print("Try to kill")
        # dataset_parsed_string = parse_feature_to_string("20190918203115156","deleteUser:0")
        # old_model.predict(dataset_parsed_string)
        # # print(old_model.prediction_thread)
        # try:
        #     old_model.prediction_thread.join()
        #     # print("CURRENT THREAD %s STATUS: %r" % (old_model.prediction_thread.getName(),old_model.prediction_thread.is_alive()))
        # except Exception as e:
        #     print(e)
        #     raise
        # # print("REMOVE current thread")
    


def load_model_for_predict():
    pickle = ACR_Pickle_Singleton.getInstance()
    acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = pickle.acr_label_encoders, pickle.articles_metadata_df,pickle.content_article_embeddings_matrix
    content_article_embeddings_matrix = min_max_scale(content_article_embeddings_matrix, min_max_range=(-0.1, 0.1))
    articles_features_config = get_articles_features_config(acr_label_encoders)
    articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)


    nar_label_encoders = \
        NAR_Pickle_Singleton.getInstance()
    session_features_config = get_session_features_config(nar_label_encoders)

    return articles_features_config, content_article_embeddings_matrix, articles_metadata, session_features_config, acr_label_encoders, articles_metadata_df
#Parse function
from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import numeric_scalers
from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import preprocess_for_predict

def parse_feature_to_string(news_id, guid):
    pickle = ACR_Pickle_Singleton.getInstance()
    acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = pickle.acr_label_encoders, pickle.articles_metadata_df, pickle.content_article_embeddings_matrix

    def get_article_text_length(article_id):
        # article_id is str
        # print("articale_id: {}".format(article_id))
        # print(articles_metadata_df.dtypes)

        if article_id == 0:
            return numeric_scalers['text_length']['avg']
        articles_metadata_df.set_index('article_id', inplace=False)
        # text_length = articles_metadata_df.loc[article_id]['text_length']
        text_length = articles_metadata_df[articles_metadata_df['article_id'] == article_id]['text_length'].values[0]
        return text_length

    dataset_parsed_string = preprocess_for_predict(guid,news_id,get_article_text_length)
    return dataset_parsed_string
#Old Prediction function
# def predict(news_id, guid):
#     try:
#         pickle = ACR_Pickle_Singleton.getInstance()
#         acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = pickle.acr_label_encoders, pickle.articles_metadata_df, pickle.content_article_embeddings_matrix

#         def get_article_text_length(article_id):
#             # article_id is str
#             # print("articale_id: {}".format(article_id))
#             # print(articles_metadata_df.dtypes)
#             from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import numeric_scalers
#             if article_id == 0:
#                 return numeric_scalers['text_length']['avg']
#             articles_metadata_df.set_index('article_id', inplace=False)
#             # text_length = articles_metadata_df.loc[article_id]['text_length']
#             text_length = articles_metadata_df[articles_metadata_df['article_id'] == article_id]['text_length'].values[0]
#             return text_length

#         # print("==================================>content_article_embeddings_matrix.shape[0]")
#         # print(content_article_embeddings_matrix.shape[0])
#         global clicked_items_state

#         # clicked_items_state = ClickedItemsState(1.0,20000, 5000, content_article_embeddings_matrix.shape[0])
#         clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_hours,
#                                                 FLAGS.recent_clicks_buffer_max_size,
#                                                 FLAGS.recent_clicks_for_normalization,
#                                                 content_article_embeddings_matrix.shape[0])
#         clicked_items_state.reset_state()
#         # model = get_estimator(model_output_dir, content_article_embeddings_matrix, articles_metadata,
#         #                 articles_features_config, session_features_config)
#         model = NAR_Model_Predict.getInstance().model
#         from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import preprocess_for_predict
#         start = time()
#         dataset_parsed_string = preprocess_for_predict(guid,news_id,get_article_text_length)
#         end = time()
#         # print("PREPROCESS TIME:"+ str(-start+end))
#         result = model.predict(dataset_parsed_string)
#         # dataset = prepare_data_for_prediction(dataset,session_features_config)
#         #a1 = lambda: prepare_data_for_prediction(dataset,session_features_config)
#         #iter_pred = model.predict(input_fn=lambda: prepare_data_for_prediction(dataset,session_features_config,truncate_session_length=FLAGS.truncate_session_length))
#         #print(iter_pred)
#         # for pred_result in iter_pred:
#         #     print(pred_result)
#         #a2 = lambda: prepare_dataset_iterator(training_files_chunk, session_features_config,
#         #                                                                  batch_size=FLAGS.batch_size,
#         #                                                                  truncate_session_length=FLAGS.truncate_session_length,predict_only=True)
#         #training_files_chunk="/home/minh/VCC/newsrecomdeepneural/nardata/tmp/test.tfrecord.gz"
#         # iter_pred2 = model.predict(input_fn=lambda: prepare_dataset_iterator(dataset_file_name, session_features_config,
#         #                                                                   batch_size=20,
#         #                                                                   truncate_session_length=FLAGS.truncate_session_length,predict_only=True))
#         end = time()
#         # for pred_result in iter_pred2:
#         #     print(pred_result)
#         # print("COUNTER:")
#         print("PREDICT TIME: %f "%(end-start))
#         # print("LIST PREDICT:")
#         # print(list(iter_pred2))
#         # print("Predict success and Return values")
#         return result
#     except Exception as ex:
#         tf.logging.error('ERROR: {}'.format(ex))
#         raise
def predict(news_id,guid):
    try:
        model = NAR_Model_Predict.getInstance()
        return model.predict_fn(model.session,news_id,guid)

    except Exception as ex:
        tf.logging.error('ERROR: {}'.format(ex))
        raise

def get_estimator(model_output_dir, content_article_embeddings_matrix, articles_metadata,
                articles_features_config, session_features_config):
    return build_estimator(model_output_dir, 
            content_article_embeddings_matrix, articles_metadata, articles_features_config,
            session_features_config)

def main(unused_argv):
    try:
        # pickle = ACR_Pickle_Singleton.getInstance()
        # model_output_dir =pickle.model_nar_dir

        # acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = pickle.acr_label_encoders, pickle.articles_metadata_df, pickle.content_article_embeddings_matrix
        # content_article_embeddings_matrix = min_max_scale(content_article_embeddings_matrix, min_max_range=(-0.1, 0.1))
        # articles_features_config = get_articles_features_config(acr_label_encoders)
        # articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)

        # nar_label_encoders = \
        #     NAR_Pickle_Singleton.getInstance()
        # session_features_config = get_session_features_config(nar_label_encoders)
        # global clicked_items_state
        # clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_hours,
        #                                         FLAGS.recent_clicks_buffer_max_size,
        #                                         FLAGS.recent_clicks_for_normalization,
        #                                         content_article_embeddings_matrix.shape[0])
        # clicked_items_state.reset_state()
        # estimator = build_estimator(model_output_dir,
        #                     content_article_embeddings_matrix, articles_metadata, articles_features_config,
        #                     session_features_config)
        
        # Old Pred, using threaded queue
        print(predict("20190926084500472","2265891616712405988"))
        # New Pred, export model -> using model to predict
        # export_path = "/home/minh/VCC/newsrecomdeepneural/nardata/exported/serving"
        # export_model(estimator, export_path)
        # export_path = "/home/minh/VCC/newsrecomdeepneural/nardata/exported/serving/1571386208"
        # a = predict_from_expoted_model("20190926084500472","ahuhu",export_path)

        # print(a)
        # start_time = time()
        # a = predict("20190926084500472","2265891616712405988")
        # end_time = time()
        # print("COUNTER FULL: "+ str(-start_time+end_time))
        # for i in range(5):
        #     print(i)
        #     print("-"*i)
        #     print(predict("20190926084500472","2265891616712405988"+str(i)))
        # print("PRELOAD: "+ str(-end_time+time()))

        # Capture whether it will be a single training job or a hyper parameter tuning job.
        tf_config_env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = tf_config_env.get('task') or {'type': 'master', 'index': 0}
        trial = task_data.get('trial')

        running_on_mlengine = (len(tf_config_env) > 0)
        print('Running {}'.format('on Google ML Engine' if running_on_mlengine else 'on a server/machine'))

        #Disabling duplicate logs on console when running locally
        logging.getLogger('tensorflow').propagate = running_on_mlengine

        tf.logging.info('Starting training job')    

        gcs_model_output_dir = FLAGS.model_dir
        #If must persist and load model ouput in a local cache (to speedup in ML Engine)
        if FLAGS.use_local_cache_model_dir:
            model_output_dir = tempfile.mkdtemp()
            tf.logging.info('Created local temp folder for models output: {}'.format(model_output_dir))
        else:
            model_output_dir = gcs_model_output_dir

        if trial is not None:
            model_output_dir = os.path.join(model_output_dir, trial)
            gcs_model_output_dir = os.path.join(gcs_model_output_dir, trial)
            tf.logging.info(
                "Hyperparameter Tuning - Trial {} - model_dir = {} - gcs_model_output_dir = {} ".format(trial, model_output_dir, gcs_model_output_dir))

        tf.logging.info('Will save temporary model outputs to {}'.format(model_output_dir))

        #If should warm start training from other previously trained model
        if FLAGS.warmup_model_dir != None:
            tf.logging.info('Copying model outputs from previous job ({}) for warm start'.format(FLAGS.warmup_model_dir))
            dowload_model_output_from_gcs(model_output_dir, 
                                          gcs_model_dir=FLAGS.warmup_model_dir,
                                          files_pattern=['graph.pb', 
                                                         'model.ckpt-', 
                                                         'checkpoint'])

            local_files_after_download_to_debug = list(glob.iglob("{}/**/*".format(model_output_dir), recursive=True))
            tf.logging.info('Files copied from GCS to warm start training: {}'.format(local_files_after_download_to_debug))

        tf.logging.info('Loading ACR module assets')
        acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = \
                load_acr_module_resources(FLAGS.acr_module_resources_path)

        #Min-max scaling of the ACR embedding for a compatible range with other input features for NAR module
        content_article_embeddings_matrix = min_max_scale(content_article_embeddings_matrix, min_max_range=(-0.1,0.1))

        articles_features_config = get_articles_features_config(acr_label_encoders)
        articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)

        
        tf.logging.info('Loading NAR module preprocesing assets')
        nar_label_encoders=load_nar_module_preprocessing_resources(FLAGS.nar_module_preprocessing_resources_path) 

        session_features_config = get_session_features_config(nar_label_encoders)
 

        tf.logging.info('Building NAR model')
        global eval_sessions_metrics_log, sessions_negative_items_log, sessions_chameleon_recommendations_log, global_eval_hour_id
        eval_sessions_metrics_log = []
        clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_hours,
                                                FLAGS.recent_clicks_buffer_max_size, 
                                                FLAGS.recent_clicks_for_normalization, 
                                                content_article_embeddings_matrix.shape[0])
        model = build_estimator(model_output_dir, 
            content_article_embeddings_matrix, articles_metadata, articles_features_config,
            session_features_config)

        
        tf.logging.info('Getting training file names')
        train_files = resolve_files(FLAGS.train_set_path_regex)

        if FLAGS.train_files_from > FLAGS.train_files_up_to:
            raise Exception('Final training file cannot be lower than Starting training file')
        train_files = train_files[FLAGS.train_files_from:FLAGS.train_files_up_to+1]

        tf.logging.info('{} files where the network will be trained and evaluated on, from {} to {}' \
                            .format(len(train_files), train_files[0], train_files[-1]))

        start_train = time()
        tf.logging.info("Starting Training Loop")
        
        training_files_chunks = list(chunks(train_files, FLAGS.training_hours_for_each_eval))

        cur_time = time()
        all_time = []
        for chunk_id in range(0, len(training_files_chunks)-1):     

            training_files_chunk = training_files_chunks[chunk_id]
            tf.logging.info('Training files from {} to {}'.format(training_files_chunk[0], training_files_chunk[-1]))
            if FLAGS.prediction_only:
                iter_pred = model.predict(input_fn=lambda: prepare_dataset_iterator(training_files_chunk, session_features_config,
                                                                      batch_size=FLAGS.batch_size,
                                                                      truncate_session_length=FLAGS.truncate_session_length,predict_only=True))

                for pred_result in iter_pred:
                    pred_time = time()
                    dt =pred_time-cur_time
                    all_time.append(dt)
                    cur_time = pred_time

                continue
            model.train(input_fn=lambda: prepare_dataset_iterator(training_files_chunk, session_features_config, 
                                                                        batch_size=FLAGS.batch_size,
                                                                        truncate_session_length=FLAGS.truncate_session_length))
            
            if chunk_id < len(training_files_chunks)-1:
                #Using the first hour of next training chunck as eval
                eval_file = training_files_chunks[chunk_id+1][0]
                tf.logging.info('Evaluating file {}'.format(eval_file))
                model.evaluate(input_fn=lambda: prepare_dataset_iterator(eval_file, session_features_config, 
                                                                                 batch_size=FLAGS.batch_size,
                                                                                 truncate_session_length=FLAGS.truncate_session_length))

            #After each number of train/eval loops
            if chunk_id % FLAGS.save_results_each_n_evals == 0:
                tf.logging.info('Saving eval metrics')
                global_eval_hour_id += 1
                save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, model_output_dir,
                                        training_hours_for_each_eval=FLAGS.training_hours_for_each_eval)

                if FLAGS.save_eval_sessions_negative_samples:
                    #Flushing to disk the negative samples used to evaluate each sessions, 
                    #so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
                    save_sessions_negative_items(model_output_dir, sessions_negative_items_log)
                    sessions_negative_items_log = []

                if FLAGS.save_eval_sessions_recommendations:  
                    #Flushing to disk the recommended items to test re-ranking approaches (e.g. MMR)
                    save_sessions_chameleon_recommendations_log(model_output_dir, 
                                sessions_chameleon_recommendations_log, global_eval_hour_id)
                    sessions_chameleon_recommendations_log = []                    

                    #Incrementing the eval hour id                    
                    global_eval_hour_id += 1
                    

                #If must persist and load model ouput in a local cache (to speedup in ML Engine)
                if FLAGS.use_local_cache_model_dir:
                    tf.logging.info('Uploading cached results to GCS')
                    upload_model_output_to_gcs(model_output_dir, gcs_model_dir=gcs_model_output_dir,  
                                               #files_pattern=None)
                                               files_pattern=[#'events.out.tfevents.', 
                                               '.csv', '.json'])

        all_time = np.array(all_time, dtype=np.float)
        print("Min %f s, max %f s, avg %f s" % (np.min(all_time), np.max(all_time), np.average(all_time)))
        print(all_time)

        tf.logging.info('Finalized Training')
        if not FLAGS.prediction_only:
            save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, model_output_dir,
                                            training_hours_for_each_eval=FLAGS.training_hours_for_each_eval)

        if FLAGS.save_eval_sessions_negative_samples:
            #Flushing to disk the negative samples used to evaluate each sessions, 
            #so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
            save_sessions_negative_items(model_output_dir, sessions_negative_items_log)

        if FLAGS.save_eval_sessions_recommendations:             
            #Flushing to disk the recommended items to test re-ranking approaches (e.g. MMR)
            save_sessions_chameleon_recommendations_log(model_output_dir, sessions_chameleon_recommendations_log, global_eval_hour_id)

        tf.logging.info('Saved eval metrics')

        #If must persist and load model ouput in a local cache (to speedup in ML Engine)
        if FLAGS.use_local_cache_model_dir:
            #Uploads all files to GCS
            upload_model_output_to_gcs(model_output_dir, gcs_model_dir=gcs_model_output_dir,
                                        files_pattern=None)
            

        log_elapsed_time(start_train, 'Finalized TRAINING Loop')
    
    except Exception as ex:
        tf.logging.error('ERROR: {}'.format(ex))
        raise



if __name__ == '__main__':  
    tf.app.run()    
