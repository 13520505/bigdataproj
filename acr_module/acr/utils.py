import tensorflow as tf
import pickle
from time import time
import numpy as np


def serialize(filename, obj):
    #with open(filename, 'wb') as handle:
    with tf.gfile.Open(filename, 'wb') as handle:
        pickle.dump(obj, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
        
def deserialize(filename):
    #with open(filename, 'rb') as handle:
    with tf.gfile.Open(filename, 'rb') as handle:
        return pickle.load(handle)

def merge_two_dicts(x, y):
    #Python 2 to 3.4
    #z = x.copy()   # start with x's keys and values
    #z.update(y)    # modifies z with y's keys and values & returns None
    #return z
    #Python 3.5 or greater
    return {**x, **y}

def log_elapsed_time(start_time, text=''):
    took = (time() - start_time) / 60.0
    tf.logging.info('==== {} elapsed {:.1f} minutes'.format(text, took))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_tf_dtype(dtype):
    if dtype == 'int':
        tf_dtype = tf.int64
    elif dtype == 'float':
        tf_dtype = tf.float32
    elif dtype == 'string' or dtype == 'bytes':
        tf_dtype = tf.string
    else:
        raise Exception('Invalid dtype "{}"'.format(dtype))
    return tf_dtype            


def resolve_files(regex):
    """Return list of files given a regex"""
    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return list(sorted(files))        

def get_pad_token():
    PAD_TOKEN = '<PAD>'
    return PAD_TOKEN
 
def get_unfrequent_token():
    UNFREQ_TOKEN = '<UNF>'
    return UNFREQ_TOKEN

def get_categ_encoder_from_values(values, include_pad_token=True, include_unfrequent_token=False):
    encoder_values = []
    if include_pad_token:
        encoder_values.append(get_pad_token())
    if include_unfrequent_token:
        encoder_values.append(get_unfrequent_token())
    encoder_values.extend(values)
    encoder_ids = list(range(len(encoder_values)))
    encoder_dict = dict(zip(encoder_values, encoder_ids))
    return encoder_dict

# def get_categ_encoder_from_values_news_id(values,  include_pad_token=True, include_unfrequent_token=False):
#     encoder_values = []
#     encoder_ids = []
#     if include_pad_token:
#         encoder_values.append(get_pad_token())
#         encoder_ids.append(0)
#     if include_unfrequent_token:
#         encoder_values.append(get_unfrequent_token())
#
#     encoder_values.extend(values)
#     encoder_ids.extend(values)
#     encoder_dict = dict(zip(encoder_values, encoder_ids))
#     return encoder_dict

def encode_categ_feature(value, encoder_dict):
    if value in encoder_dict:
        return encoder_dict[value]
    else:
        return encoder_dict[get_unfrequent_token()]

# def encode_categ_feature_second_time(value, encoder_dict):
#     if value in encoder_dict:
#         return encoder_dict[value]
#     else:
#         return


# tung
def get_categ_encoder_from_values_second_time_category(values, acr_label_encoders_values):
    encoder_keys = []
    encoder_values = []
    for i in range(0, len(values)):
        if values[i] in acr_label_encoders_values.keys():
            encoder_keys.append(values[i])
            encoder_values.append(acr_label_encoders_values[values[i]])
        else:
            encoder_keys.append(values[i])
            encoder_values.append(acr_label_encoders_values[get_pad_token()])
    encoder_dict = dict(zip(encoder_keys, encoder_values))
    return encoder_dict

    # print("Load dictionary cate")
    # pickle_in = open("./acr_module/config/dict_cat.pickle", "rb")
    # example_dict_a = pickle.load(pickle_in)
    #
    # if not example_dict_a: #dict is empty
    #     encoder_values = []
    #     encoder_values.extend(values)
    #     encoder_ids = list(range(len(encoder_values)))
    #     encoder_dict = dict(zip(encoder_values, encoder_ids))
    #     return encoder_dict
    # else:
    #     list_cat_in_dict = example_dict_a.keys()
    #
    #     list_after = values
    #     list_new = [i for i in list_after if i not in list_cat_in_dict]
    #
    #     encoder_values = []
    #
    #     if not list_new:  # check if list empty
    #         encoder_dict = {}
    #         return encoder_dict
    #     else:
    #         encoder_values.extend(list_new)
    #         encoder_ids = list(range(len(encoder_values)))
    #         encoder_ids = list(np.asarray(encoder_ids) + max(example_dict_a.values()) + 1)
    #         encoder_dict = dict(zip(encoder_values, encoder_ids))
    #         return encoder_dict

# def get_categ_encoder_from_values_second_time(values, acr_label_encoders_values ):
#     encoder_keys = []
#     encoder_values = []
#     for i in range(0, len(values)):
#         if values[i] in acr_label_encoders_values:
#             encoder_keys.append(values[i])
#             encoder_values.append(acr_label_encoders_values[values[i]])
#     encoder_dict = dict(zip(encoder_keys, encoder_values))
#     return encoder_dict

def get_categ_encoder_from_values_second_time_news_id(values, acr_label_encoders_values):
    encoder_values = []
    encoder_values.extend(values)
    print(values)

    encoder_ids = list(range(len(values)))
    encoder_ids = list(np.asarray(encoder_ids) + max(acr_label_encoders_values.values()) + 1)
    encoder_dict = dict(zip(encoder_values, encoder_ids))
    return encoder_dict


