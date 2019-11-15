import logging
import os
from time import time
import json
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.preprocessing import StandardScaler
import sys
# sys.path.append("/data1/tungtv/code/chameleon/newsrecomdeepneural")


from acr_module.acr.acr_module_service import get_all_file, split_date, load_json_config
from acr_module.acr.utils import resolve_files, deserialize, serialize, log_elapsed_time
from acr_module.acr.acr_model import ACR_Model
from acr_module.acr.acr_datasets import prepare_dataset
import os.path
from os import path
import mysql.connector
import pickle
import operator
from tensorflow.contrib import predictor

from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import delete_all_file_in_path

tf.logging.set_verbosity(tf.logging.INFO)

RANDOM_SEED=42

#Control params
#tf.flags.DEFINE_string('data_dir', default='',
#                    help='Directory where the dataset is located')
tf.flags.DEFINE_string('train_set_path_regex',
                    default='/train*.tfrecord', help='Train set regex')
tf.flags.DEFINE_string('model_dir', default='./tmp',
                    help='Directory where save model checkpoints')

tf.flags.DEFINE_string('input_word_vocab_embeddings_path', default='',
                    help='Input path for a pickle with words vocabulary and corresponding word embeddings')
tf.flags.DEFINE_string('input_label_encoders_path', default='',
                    help='Input path for a pickle with label encoders (article_id, category_id, publisher_id)')
tf.flags.DEFINE_string('output_acr_metadata_embeddings_path', default='',
                    help='Output path for a pickle with articles metadata and content embeddings')

#Model params
tf.flags.DEFINE_string('text_feature_extractor', default="CNN", help='Feature extractor of articles text: CNN or RNN')
tf.flags.DEFINE_string('training_task', default="metadata_classification", help='Training task: (metadata_classification | autoencoder)')
tf.flags.DEFINE_float('autoencoder_noise', default=0.0, help='Adds white noise with this standard deviation to the input word embeddings')


tf.flags.DEFINE_string('cnn_filter_sizes', default="3,4,5", help='CNN layers filter sizes (sliding window over words)')
tf.flags.DEFINE_integer('cnn_num_filters', default=128, help='Number of filters of CNN layers')

tf.flags.DEFINE_integer('rnn_units', default=250, help='Number of units in each RNN layer')
tf.flags.DEFINE_integer('rnn_layers', default=1, help='Number of RNN layers')
tf.flags.DEFINE_string('rnn_direction', default='unidirectional', help='Direction of RNN layers: (unidirectional | bidirectional)')



tf.flags.DEFINE_integer('acr_embeddings_size', default=250, help='Embedding size of output ACR embeddings')


#Training params
tf.flags.DEFINE_integer('batch_size', default=64, help='Batch size')
tf.flags.DEFINE_integer('training_epochs', default=10, help='Training epochs')
tf.flags.DEFINE_float('learning_rate', default=1e-3, help='Lerning Rate')
tf.flags.DEFINE_float('dropout_keep_prob', default=1.0, help='Dropout (keep prob.)')
tf.flags.DEFINE_float('l2_reg_lambda', default=1e-3, help='L2 regularization')


FLAGS = tf.flags.FLAGS
#params_dict = tf.app.flags.FLAGS.flag_values_dict()
#tf.logging.info('PARAMS: {}'.format(json.dumps(params_dict)))


def get_session_features_config(acr_label_encoders):
    features_config = {
                'single_features':
                    {'article_id': {'type': 'categorical', 'dtype': 'int'},
                    'category0': {'type': 'categorical', 'dtype': 'int'},
                    # 'category1': {'type': 'categorical', 'dtype': 'int'},
                    # 'author': {'type': 'categorical', 'dtype': 'int'},
                    'created_at_ts': {'type': 'numerical', 'dtype': 'int'},
                    'text_length': {'type': 'numerical', 'dtype': 'int'},
                    },
                'sequence_features': {
                    'text': {'type': 'numerical', 'dtype': 'int'},
                    'keywords': {'type': 'categorical', 'dtype': 'int'},
                    # 'concepts': {'type': 'categorical', 'dtype': 'int'},
                    # 'entities': {'type': 'categorical', 'dtype': 'int'},
                    'locations': {'type': 'categorical', 'dtype': 'int'},
                    'persons': {'type': 'categorical', 'dtype': 'int'},
                    },
                'label_features': {
                    'category0': {'type': 'categorical', 'dtype': 'int', 'classification_type': 'multiclass', 'feature_weight_on_loss': 1.0},
                    ## 'category1': {'type': 'categorical', 'dtype': 'int', 'classification_type': 'multiclass'}, #Too unbalanced
                    'keywords': {'type': 'categorical', 'dtype': 'int', 'classification_type': 'multilabel', 'feature_weight_on_loss': 1.0},
                    }
                }


    #Adding cardinality to categorical features
    for feature_groups_key in features_config:
        features_group_config = features_config[feature_groups_key]
        for feature_name in features_group_config:
            if feature_name in acr_label_encoders and features_group_config[feature_name]['type'] == 'categorical':
                features_group_config[feature_name]['cardinality'] = len(acr_label_encoders[feature_name])

    tf.logging.info('Session Features: {}'.format(features_config))

    return features_config


def load_acr_preprocessing_assets(acr_label_encoders_path, word_vocab_embeddings_path):
    (acr_label_encoders, labels_class_weights) = deserialize(acr_label_encoders_path)
    article_id_encoder = acr_label_encoders['article_id']
    tf.logging.info("Read article id label encoder: {}".format(len(acr_label_encoders['article_id'])))

    tf.logging.info("Classes weights available for: {}".format(labels_class_weights.keys()))

    (word_vocab, word_embeddings_matrix) = deserialize(word_vocab_embeddings_path)
    tf.logging.info("Read word embeddings: {}".format(word_embeddings_matrix.shape))

    return acr_label_encoders, labels_class_weights, word_embeddings_matrix

def create_multihot_feature(features, column_name, features_config):
    column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
            key=column_name, num_buckets=features_config['sequence_features'][column_name]['cardinality']))
    return column


PREDICTIONS_PREFIX = "predictions-"
def acr_model_fn(features, labels, mode, params):

    #keywords_column  = create_multihot_feature(features, 'keywords', params['features_config'])
    # concepts_column  = create_multihot_feature(features, 'concepts', params['features_config'])
    # entities_column  = create_multihot_feature(features, 'entities', params['features_config'])
    locations_column = create_multihot_feature(features, 'locations', params['features_config'])
    persons_column   = create_multihot_feature(features, 'persons', params['features_config'])

    # metadata_input_feature_columns = [concepts_column, entities_column, locations_column, persons_column]
    metadata_input_feature_columns = [locations_column, persons_column]

    metadata_input_features = {#'concepts': features['concepts'],
                                # 'entities': features['entities'],
                               'locations': features['locations'],
                               'persons': features['persons']}



    acr_model = ACR_Model(params['training_task'], params['text_feature_extractor'], features, metadata_input_features,
                        metadata_input_feature_columns,
                         labels, params['features_config']['label_features'],
                         mode, params)

    loss = None
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        loss = acr_model.total_loss

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = acr_model.train_op

    eval_metrics = {}
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        eval_metrics = acr_model.eval_metrics

    predictions = None
    prediction_hooks = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {#Category prediction
                       #'predicted_category_id': acr_model.predictions,
                       #Trained ACR embeddings
                       'acr_embedding': acr_model.article_content_embedding,
                       #Additional metadata
                       'article_id': features['article_id'],
                       'category0': features['category0'],
                       # 'category1': features['category1'],
                       # 'author': features['author'],
                       'keywords': features['keywords'],
                       # 'concepts': features['concepts'],
                       # 'entities': features['entities'],
                       'locations': features['locations'],
                       'persons': features['persons'],
                       'created_at_ts': features['created_at_ts'],
                       'text_length': features['text_length'],
                       'input_text': features['text']
                       }

        if params['training_task'] == 'autoencoder':
            #predictions['input_text'] = features['text']
            predictions['predicted_word_ids'] = acr_model.predicted_word_ids
        elif params['training_task'] == 'metadata_classification':
            #Saves predicted categories
            for feature_name in acr_model.labels_predictions:
                predictions["{}{}".format(PREDICTIONS_PREFIX, feature_name)] = acr_model.labels_predictions[feature_name]

        #prediction_hooks = [ACREmbeddingExtractorHook(mode, acr_model)]

    training_hooks = []
    if params['enable_profiler_hook']:
        profile_hook = tf.train.ProfilerHook(save_steps=100,
                                    save_secs=None,
                                    show_dataflow=True,
                                    show_memory=False)
        training_hooks=[profile_hook]


    return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              loss=loss,
              train_op=train_op,
              eval_metric_ops=eval_metrics,
              training_hooks=training_hooks
              #prediction_hooks=prediction_hooks,
              )


def build_acr_estimator(model_output_dir, word_embeddings_matrix, features_config, labels_class_weights, special_token_embedding_vector, list_args):


    params = {'training_task': list_args["training_task"],
              'text_feature_extractor': list_args["text_feature_extractor"],
              'word_embeddings_matrix': word_embeddings_matrix,
              'vocab_size': word_embeddings_matrix.shape[0],
              'word_embedding_size': word_embeddings_matrix.shape[1],
              'cnn_filter_sizes': list_args["cnn_filter_sizes"],
              'cnn_num_filters': list_args["cnn_num_filters"],
              'rnn_units': list_args["rnn_units"],
              'rnn_layers': list_args["rnn_layers"],
              'rnn_direction': list_args["rnn_direction"],
              'dropout_keep_prob': list_args["dropout_keep_prob"],
              'l2_reg_lambda': list_args["l2_reg_lambda"],
              'learning_rate': list_args["learning_rate"],
              'acr_embeddings_size': list_args["acr_embeddings_size"],
              'features_config': features_config,
              'labels_class_weights': labels_class_weights,
              'special_token_embedding_vector': special_token_embedding_vector,
              'autoencoder_noise': list_args["autoencoder_noise"],
              'enable_profiler_hook': False
              }

    session_config = tf.ConfigProto(allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        save_summary_steps=50,
                                        keep_checkpoint_max=1,
                                        session_config=session_config
                                       )

    acr_cnn_classifier = tf.estimator.Estimator(model_fn=acr_model_fn,
                                            model_dir=model_output_dir,
                                            params=params,
                                            config=run_config)

    return acr_cnn_classifier


def export_acr_metadata_embeddings(acr_label_encoders, articles_metadata_df, content_article_embeddings):
    output_path = FLAGS.output_acr_metadata_embeddings_path
    tf.logging.info('Exporting ACR Label Encoders, Article metadata and embeddings to {}'.format(output_path))
    to_serialize = (acr_label_encoders, articles_metadata_df, content_article_embeddings)
    serialize(output_path, to_serialize)


def get_articles_metadata_embeddings(article_metadata_with_pred_embeddings):
    articles_metadata_df = pd.DataFrame(article_metadata_with_pred_embeddings).sort_values(by='article_id')

    tf.logging.info("First article id: {}".format(articles_metadata_df['article_id'].head(1).values[0]))
    tf.logging.info("Last article id: {}".format(articles_metadata_df['article_id'].tail(1).values[0]))

    #Checking whether article ids are sorted and contiguous
    # assert (articles_metadata_df['article_id'].head(1).values[0] == 1) #0 is reserved for padding
    # assert (len(articles_metadata_df) == articles_metadata_df['article_id'].tail(1).values[0])

    content_article_embeddings = np.vstack(articles_metadata_df['acr_embedding'].values)


    # #Standardizing the Article Content Embeddings for Adressa dataset, and scaling to get maximum and minimum values around [-6,5], instead of [-40,30] after standardization, to mimic the doc2vec distribution for higher accuracy in NAR module
    # scaler = StandardScaler()
    # content_article_embeddings_standardized = scaler.fit_transform(content_article_embeddings)
    # content_article_embeddings_standardized_scaled = content_article_embeddings_standardized / 5.0
    #
    #
    #Creating and embedding for the padding article
    embedding_for_padding_article = np.mean(content_article_embeddings, axis=0)
    content_article_embeddings_with_padding = np.vstack([embedding_for_padding_article, content_article_embeddings])

    #Checking if content articles embedding size correspond to the last article_id
    # assert content_article_embeddings_with_padding.shape[0] == articles_metadata_df['article_id'].tail(1).values[0]+1

    #Converting keywords multi-label feature from multi-hot representation back to list of keyword ids
    preds_keywords_column_name = "{}{}".format(PREDICTIONS_PREFIX, "keywords")
    if preds_keywords_column_name in articles_metadata_df.columns:
        articles_metadata_df[preds_keywords_column_name] = articles_metadata_df[preds_keywords_column_name] \
                                                            .apply(lambda x: x.nonzero()[0])



    # cols_to_export = ['article_id', 'category0', 'category1',
    #                    'author', 'keywords', 'concepts', 'entities', 'locations', 'persons',
    #                    'created_at_ts', 'text_length', 'input_text']

    cols_to_export = ['article_id', 'category0', 'keywords', 'locations', 'persons',
                      'created_at_ts', 'text_length', 'input_text']

    if FLAGS.training_task == 'autoencoder':
        cols_to_export.extend(['predicted_word_ids'])
    elif FLAGS.training_task == 'metadata_classification':
        #Adding predictions columns for debug
        cols_to_export.extend([col for col in articles_metadata_df.columns if col.startswith(PREDICTIONS_PREFIX)])


    #Filtering metadata columns to export
    # articles_metadata_df = articles_metadata_df[['article_id', 'category0', 'category1',
    #                                              'author', 'keywords', 'concepts', 'entities', 'locations', 'persons',
    #                                              'created_at_ts', 'text_length'] + \
    #                                              list([column_name for column_name in articles_metadata_df.columns \
    #                                                    if column_name.startswith(PREDICTIONS_PREFIX)])] #Adding predictions columns for debug

    # articles_metadata_df = articles_metadata_df[['article_id', 'category0',  'keywords',  'locations', 'persons',
    #                                              'created_at_ts', 'text_length'] + \
    #                                             list([column_name for column_name in articles_metadata_df.columns \
    #                                                   if column_name.startswith(
    #                                                     PREDICTIONS_PREFIX)])]  # Adding predictions columns for debug
    articles_metadata_df = articles_metadata_df[cols_to_export]


    return articles_metadata_df, content_article_embeddings_with_padding

def Merge(dict1, dict2):
    return(dict2.update(dict1))

def export_acr_metadata_embeddings_with_datetime(df, acr_label_encoders, articles_metadata_df, content_article_embeddings, path):

    # if os.path.exists('./acr_module/config/dict_news_id_encode.pickle'):
    #     pickle_in = open("./acr_module/config/dict_news_id_encode.pickle", "rb")
    #     dict_id_old = pickle.load(pickle_in)
    #
    #     list_key_old = list(dict_id_old.keys())
    #     list_value_old = list(dict_id_old.values())
    #
    #     dict_id_new = acr_label_encoders['article_id']
    #     list_key_new = list(dict_id_new.keys())
    #     list_value_new = list(dict_id_new.values())
    #
    #     list_key_old = list_key_old.append(list_key_new)
    #     list_value_old = list_value_old.append(list_value_new)
    #
    #     acr_label_encoders['article_id'] = dict(zip(list_key_old, list_value_old ))
    #
    #     # Merge(dict_id_old, dict_id_new)
    #     #
    #     # acr_label_encoders['article_id'] = dict_id_new
    #     # acr_label_encoders['article_id'] = sorted(dict_id_new.items(), key=operator.itemgetter(1))
    #     # acr_label_encoders['article_id'] = dict(acr_label_encoders['article_id'])


    if len(os.listdir( path)) != 0 :
        (acr_label_encoders_load, articles_metadata_df_load, content_article_embeddings_load) = deserialize((get_all_file( path))[0])

        # append article_id
        list_key_old = list(df['id'])
        list_key_old.insert(0, "<PAD>")
        list_value_old = list(df['id_encoded'])
        list_value_old.insert(0, 0)

        # list_value_old = list(acr_label_encoders_load['article_id'].values())

        # list_key_old.extend(list(acr_label_encoders['article_id'].keys()))
        # list_value_old.extend(list(acr_label_encoders['article_id'].values()))
        #
        # print("new and old acr_label_encoders = >>>>>>>>>>")
        # print(len(list(acr_label_encoders['article_id'].keys())))
        # print(len(list(acr_label_encoders['article_id'].values())))
        #
        # print(len(list_key_old))
        # print(len(list_value_old))


        acr_label_encoders['article_id'] = dict(zip(list_key_old, list_value_old))
        print(len(acr_label_encoders['article_id'].keys()))
        print(len(acr_label_encoders['article_id'].values()))

        # append df
        print("load : {}".format(len(articles_metadata_df_load)))
        print("df new : {}".format(len(articles_metadata_df)))
        frames = [articles_metadata_df_load, articles_metadata_df]
        articles_metadata_df = pd.concat(frames)
        articles_metadata_df = articles_metadata_df.reset_index()
        articles_metadata_df = articles_metadata_df.drop(columns='index')
        # articles_metadata_df = articles_metadata_df.set_index('article_id')

        # append content_art_embeding
        print("load matrix : {}".format(len(content_article_embeddings_load)))
        print("df new : {}".format(len(content_article_embeddings)))
        content_article_embeddings = content_article_embeddings[1:]
        content_article_embeddings =np.concatenate((content_article_embeddings_load, content_article_embeddings), axis=0)

    output_path = path + "acr_articles_metadata_embeddings.pickle"
    tf.logging.info('Exporting ACR Label Encoders, Article metadata and embeddings to {}'.format(output_path))
    to_serialize = (acr_label_encoders, articles_metadata_df, content_article_embeddings)
    serialize(output_path, to_serialize)


def save_to_mysql_database(mysql_host, mysql_user, mysql_passwd, mysql_database, acr_label_encoders,
                           articles_metadata_df, content_article_embeddings):

    '''
        -
        - database and table have already create
    '''
    mydb = mysql.connector.connect(
        host=mysql_host,
        user=mysql_user,
        passwd=mysql_passwd,
        database=mysql_database
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO customers (news_id, news_id_encode, word_embedding) VALUES (%s, %s, %s)"
    tupel = ()
    # for i in range(0, len(acr_label_encoders[0]['article_id'])):
    #     tupel = tupel + (list(acr_label_encoders[0]['article_id'].keys())[i], list(acr_label_encoders[0]['article_id'].values())[i], list(aa[2][i]))

    mycursor.execute(sql, tupel)

    mydb.commit()


from datetime import datetime


# GET CURRENT TIME
def get_date_time_current():
    now = datetime.now()
    timestamp = int(datetime.timestamp(now))
    return str(timestamp)


# SUBTRACT TIME
def subtract_month(current_time):
    from datetime import datetime
    dt_object = datetime.fromtimestamp(int(current_time))
    a = dt_object.strftime('%Y-%m-%d')

    import datetime
    import dateutil.relativedelta

    d = datetime.datetime.strptime(a, "%Y-%m-%d")
    d2 = d - dateutil.relativedelta.relativedelta(days=7)
    #     print(d2)

    from datetime import datetime
    timestamp = datetime.timestamp(d2)
    return int(timestamp)


# REMOVE PICKLE
parameter = load_json_config("./parameter.json")
def remove_acr_pickle(path_file):
    acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = deserialize(path_file)

    def serialize(filename, obj):
        # with open(filename, 'wb') as handle:
        with tf.gfile.Open(filename, 'wb') as handle:
            pickle.dump(obj, handle)

    def merge_two_dicts(x, y):
        return {**x, **y}

    # articles_metadata_df = articles_metadata_df[
    #     articles_metadata_df['created_at_ts'] >= subtract_month(get_date_time_current())]
    lena = 600

    #  acr_label_encoders
    a = acr_label_encoders["article_id"]
    a = merge_two_dicts({"<PAD>": 0}, dict(list(a.items())[-lena:]))
    acr_label_encoders["article_id"] = a

    list_key = list(acr_label_encoders["article_id"].keys())
    list_value = list(range(len(list_key)))
    acr_label_encoders["article_id"] = dict(zip(list_key, list_value))

    # df
    list_value = list_value[1:]
    articles_metadata_df = articles_metadata_df[-lena:]
    articles_metadata_df['article_id'] = list_value

    # matrix
    matrix = np.insert(content_article_embeddings_matrix[-lena:], 0, content_article_embeddings_matrix[0], axis=0)

    to_serialize = (acr_label_encoders, articles_metadata_df, matrix)

    # creat folder acr predict
    dir = "/pickles/acr_articles_metadata_embeddings_predict/"
    DATA_DIR = parameter["DATA_DIR"]
    path_predict  = DATA_DIR + dir
    from os import path
    if path.exists(path_predict):
        pass
    else:
        os.makedirs(path_predict)

    serialize(path_predict+"acr_articles_metadata_embeddings_predict.pickle", to_serialize)
def main_acr_train():
# def main(unused_argv):
    # try:
        print("<=== STARTING ARC TRAINING ===>")


        parameter = load_json_config("./parameter.json")
        list_args = parameter["acr_training"]

        DATA_DIR = parameter["DATA_DIR"]
        model_dir = DATA_DIR + list_args["model_dir"]
        train_set_path_regex = DATA_DIR + list_args["train_set_path_regex"]
        input_word_vocab_embeddings_path = DATA_DIR + list_args["input_word_vocab_embeddings_path"]
        input_label_encoders_path = DATA_DIR + list_args["input_label_encoders_path"]
        output_acr_metadata_embeddings_path = DATA_DIR + list_args["output_acr_metadata_embeddings_path"]
        batch_size = list_args["batch_size"]
        truncate_tokens_length = list_args["truncate_tokens_length"]
        training_epochs = list_args["training_epochs"]
        learning_rate = list_args["learning_rate"]
        dropout_keep_prob = list_args["dropout_keep_prob"]
        l2_reg_lambda = list_args["l2_reg_lambda"]
        text_feature_extractor = list_args["text_feature_extractor"]
        training_task = list_args["training_task"]
        cnn_filter_sizes = list_args["cnn_filter_sizes"]
        cnn_num_filters = list_args["cnn_num_filters"]
        rnn_units = list_args["rnn_units"]
        rnn_layers = list_args["rnn_layers"]
        rnn_direction = list_args["rnn_direction"]
        acr_embeddings_size = list_args["acr_embeddings_size"]
        # mysql_host = list_args["mysql_host"]
        # mysql_user = list_args["mysql_user"]
        # mysql_passwd = list_args["mysql_passwd"]
        # mysql_database = list_args["mysql_database"]




        # Capture whether it will be a single training job or a hyper parameter tuning job.
        tf_config_env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = tf_config_env.get('task') or {'type': 'master', 'index': 0}
        trial = task_data.get('trial')

        running_on_mlengine = (len(tf_config_env) > 0)
        tf.logging.info('Running {}'.format('on Google ML Engine' if running_on_mlengine else 'on a server/machine'))

        #Disabling duplicate logs on console when running locally
        logging.getLogger('tensorflow').propagate = running_on_mlengine


        start_train = time()
        tf.logging.info('Starting training job')

        model_output_dir = model_dir

        if trial is not None:
            model_output_dir = os.path.join(model_output_dir, trial)
            tf.logging.info(
                "Hyperparameter Tuning - Trial {}. model_dir = {}".format(trial, model_output_dir))
        else:
            tf.logging.info('Saving model outputs to {}'.format(model_output_dir))

        tf.logging.info('Loading ACR preprocessing assets')

        # check exist path
        if path.exists(model_output_dir):
            pass
        else:
            os.makedirs(model_output_dir)


        if path.exists(output_acr_metadata_embeddings_path):
            pass
        else:
            os.makedirs(output_acr_metadata_embeddings_path)

        print("Loading ACR preprocessing assets....")

        print(input_label_encoders_path)
        print(output_acr_metadata_embeddings_path)

        file_lable_encode = get_all_file(input_label_encoders_path)[0]
        file_word_embedding = get_all_file(input_word_vocab_embeddings_path)[0]

        # current_time = split_date(file_lable_encode)
        # print(current_time)

        # load file with max date
        acr_label_encoders, labels_class_weights, word_embeddings_matrix = \
            load_acr_preprocessing_assets(file_lable_encode,file_word_embedding)


        features_config = get_session_features_config(acr_label_encoders)

        #input_tfrecords = os.path.join(FLAGS.data_dir, FLAGS.train_set_path_regex)
        input_tfrecords = train_set_path_regex
        tf.logging.info('Defining input data (TFRecords): {}'.format(input_tfrecords))


        #Creating an ambedding for a special token to initiate decoding of RNN-autoencoder
        special_token_embedding_vector = np.random.uniform(low=-0.04, high=0.04,
                                                 size=[1,word_embeddings_matrix.shape[1]])



        # train_files = get_listmax_date(get_all_file(train_set_path_regex))
        train_files = get_all_file(train_set_path_regex)
        print(train_files)

        if len(os.listdir(model_dir)) == 0:   #acr_model not exist
            print("NO Have ACR Module")
            acr_model = build_acr_estimator(model_output_dir,
                                            word_embeddings_matrix,
                                            features_config,
                                            labels_class_weights,
                                            special_token_embedding_vector, list_args)
            tf.logging.info('Training model')
            acr_model.train(input_fn=lambda: prepare_dataset(files=train_files,
                                                  features_config=features_config,
                                                  batch_size=batch_size,
                                                  epochs=training_epochs,
                                                  shuffle_dataset=True,
                                                  shuffle_buffer_size=10000))
        else:  #acr_model   exist
            print("Have ACR Module")
            acr_model = build_acr_estimator(model_output_dir,
                                            word_embeddings_matrix,
                                            features_config,
                                            labels_class_weights,
                                            special_token_embedding_vector, list_args)

        #The objective is to overfitting this network, so that the ACR embedding represent well the articles content
        tf.logging.info('Evaluating model - TRAIN SET')
        print("Evaluating model - TRAIN SET")
        eval_results = acr_model.evaluate(input_fn=lambda: prepare_dataset(files=train_files,
                                                    features_config=features_config,
                                                    batch_size=batch_size,
                                                    epochs=1,
                                                    shuffle_dataset=False))
        tf.logging.info('Evaluation results with TRAIN SET (objective is to overfit): {}'.format(eval_results))

        '''
        tf.logging.info('Evaluating model - TEST SET')
        eval_results = acr_model.evaluate(input_fn=lambda: prepare_dataset(files=test_files,
                                                    features_config=features_config,
                                                    batch_size=FLAGS.batch_size, 
                                                    epochs=1, 
                                                    shuffle_dataset=False))
        tf.logging.info('Evaluation results with TEST SET: {}'.format(eval_results))
        '''

        tf.logging.info('Predicting ACR embeddings')
        print("Predicting ACR embeddings")
        article_metadata_with_pred_embeddings = acr_model.predict(input_fn=lambda: prepare_dataset(files=train_files,
                                                    features_config=features_config,
                                                    batch_size=batch_size,
                                                    epochs=1, 
                                                    shuffle_dataset=False))
        

        articles_metadata_df, content_article_embeddings = get_articles_metadata_embeddings(article_metadata_with_pred_embeddings)
        tf.logging.info('Generated ACR embeddings: {}'.format(content_article_embeddings.shape))   

        # read csv preprocessed by acr preprocessing
        list_args2 = parameter["acr_preprocess"]
        path_csv =  DATA_DIR + list_args2['output_articles_csv_path_preprocessed']
        df = pd.read_csv(get_all_file(path_csv)[0])
        print(len(df['id']))

        export_acr_metadata_embeddings_with_datetime(df, acr_label_encoders, articles_metadata_df, content_article_embeddings, output_acr_metadata_embeddings_path)

        print("Export done, Call load acr auto ...")

        print("Remove acr embedding")
        remove_acr_pickle(get_all_file(output_acr_metadata_embeddings_path)[0])

        # TODO gọi service load acr_label_encoders, articles_metadata_df, content_article_embeddings vào biến singleton
        # import requests
        # resp = requests.get('http://0.0.0.0:8082/loadacr')
        # if resp.status_code == 200:
        #     print("Called load acr_pickle")
        # else:
        #     print("Not Yet call load acr_pickle")

        # save_to_mysql_database( mysql_host,  mysql_user,  mysql_passwd, mysql_database, acr_label_encoders,articles_metadata_df , content_article_embeddings)

        # after trainning, delete all file tfrecord
        delete_all_file_in_path(train_set_path_regex)
        log_elapsed_time(start_train, 'Finalized TRAINING')
        print("<=== END ARC TRAINING ===>")
    
    # except Exception as ex:
    #     tf.logging.error('ERROR: {}'.format(ex))
    #     raise

if __name__ == '__main__':
    tf.app.run()
