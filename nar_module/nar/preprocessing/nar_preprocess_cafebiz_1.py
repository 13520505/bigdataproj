import os
import findspark
import hashlib
import matplotlib
# %matplotlib inline
import pyspark
import datetime
import pickle
import pandas as pd
import sys
import os.path
from os import path
import tensorflow as tf

from acr_module.acr.preprocessing.acr_preprocess_cafebiz import get_date_time_current

# sys.path.append("/data1/tungtv/code/chameleon/newsrecomdeepneural")


from acr_module.acr.acr_module_service import handle_database_news, load_json_config, \
    get_all_file, read_dbnews_mysql
from pyspark.sql.functions import col
# findspark.init()
from pyspark import SparkContext, SparkConf,SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
import warnings

from nar_module.nar.nar_trainer_cafebiz import split_string_train_path
from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import delete_all_file_in_path

warnings.filterwarnings('ignore')
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import types as t

from datetime import timedelta
from scipy import stats
from pyspark.sql.functions import to_timestamp
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, LongType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from pyspark.sql import functions as f




def spark_inital():
    # spark = SparkSession.builder.master('local[30]').appName('myAppName').config("spark.local.dir", "/data/tungtv/tmp/").getOrCreate()
    # spark = SparkSession.builder.master('yarn').appName('NAR Preprocess').config("spark.hadoop.yarn.resourcemanager.address", "10.5.36.95:8032").config("spark.local.dir","/data1/tungtv/tmp/").getOrCreate()
    # spark = SparkSession.builder.master('local[*]').appName('myAppName').getOrCreate()
    spark = SparkSession.builder.master('local[5]').appName('NAR Preprocess').config("spark.local.dir","/data1/tungtv/tmp/").config("spark.driver.memory", "12g").getOrCreate()
    # data: 69
    # data1: 95
    return spark

from dateutil.relativedelta import *
def list_date_add_month(start_date, end_date, hdfs_path):

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    date = start_date
    list_date = [hdfs_path + str(start_date).replace(str(start_date)[-3:], "-**")+'/']
    while 1:
        date = date + relativedelta(months=+1)
        if date <= end_date:
            string = str(date).replace(str(date)[-3:], "-**" )
            list_date.append(hdfs_path+string+'/')
        else:
            break
    print(list_date)
    return list_date

def handle_database_log(domain, spark, input_path_data_log_click, date_start, date_end):
    path_hdfs = input_path_data_log_click

    list_path = list_date_add_month(date_start, date_end, path_hdfs)
    # list_path = get_list_hdfs_date(date_start,date_end,path_hdfs)

    df=spark.read.option("path_hdfs",path_hdfs).parquet(*list_path)
    # df = spark.read.parquet("hdfs://10.5.37.76:8020/Data/Logging/pvTosFull/pc/2019-07-03")
    df = df.filter(col("domain").isin([domain]))
    df = df.withColumn('full_url', concat(col("domain"), col("path")))
    df = df.filter("guid != '-1'")

    df = df.withColumn("time", F.unix_timestamp(col("dt"), 'yyyy-MM-dd HH:mm:ss'))
    print(df.count())
    return df


# Retrives article id from its cannonical URL (because sometimes article ids in interactions do no match with articles tables, but cannonical URL do)
def get_article_id_encoded_from_url(canonical_url, valid_articles_urls_to_ids_dict_broadcast):
    if canonical_url in valid_articles_urls_to_ids_dict_broadcast.value:
        return valid_articles_urls_to_ids_dict_broadcast.value[canonical_url]
    return None


def add_column_dataframe(dataframe, columnName, expression):
    dataframe = dataframe.withColumn(columnName, expression)
    return dataframe


def get_timestamp_from_date_str(value):
    if value is not None:
        return int(datetime.datetime.strptime(value, 'yyyy-MM-dd HH:mm:ss').timestamp())
    return None


def check_numm(df, name_col):
    return df.where(F.isnull(F.col(name_col))).count()


# Processing categorical features
def get_categ_features_counts_dataframe(interactions_spark_df, column_name):
    df_pandas = interactions_spark_df.groupBy(column_name).count().toPandas().sort_values('count', ascending=False)
    return df_pandas


def get_encoder_for_values(values, PAD_TOKEN='<PAD>', UNFREQ_TOKEN='<UNF>'):
    encoder_values = [PAD_TOKEN, UNFREQ_TOKEN] + values
    encoder_ids = list(range(len(encoder_values)))
    encoder_dict = dict(zip(encoder_values, encoder_ids))
    return encoder_dict


def get_categ_features_encoder_dict(counts_df, min_freq=100):
    freq_values = counts_df[counts_df['count'] >= 100][counts_df.columns[0]].values.tolist()
    encoder_dict = get_encoder_for_values(freq_values)
    return encoder_dict


def encode_cat_feature(value, encoder_dict, UNFREQ_TOKEN='<UNF>'):
    if value in encoder_dict:
        return encoder_dict[value]
    else:
        return encoder_dict[UNFREQ_TOKEN]


def hash_str_to_int(encoded_bytes_text, digits):
    return int(str(int(hashlib.md5(encoded_bytes_text).hexdigest()[:8], 16))[:digits])


def close_session(session, first_timestamp_ts):
    size = len(session)

    # Creating and artificial session id based on the first click timestamp and a hash of user id
    first_click = session[0]
    session_id = (int(first_click['timestamp']) * 100) + hash_str_to_int(first_click['user_id'].encode(), 3)
    session_hour = int((first_click['timestamp'] - first_timestamp_ts) / (
                1000 * 60 * 60))  # Converting timestamp to hours since first timestamp

    # Converting to Spark DataFrame Rows, to convert RDD back to DataFrame
    # TODO add 'view' here
    clicks = list([T.Row(**click) for click in session])
    session_dict = {'session_id': session_id,
                    'session_hour': session_hour,
                    'session_size': size,
                    'session_start': first_click['timestamp'],
                    'user_id': first_click['user_id'],
                    'clicks': clicks
                    }
    session_row = T.Row(**session_dict)

    return session_row


def transform_interaction(interaction, encoders_dict):
    return {
        'article_id': interaction['article_id'],
        'url': interaction['full_url'],
        'user_id': interaction['user_id'],
        'timestamp': interaction['time'] * 1000,  # converting to timestamp
        'active_time_secs': interaction['top'],
        #             'country': encode_cat_feature(interaction['country'], encoders_dict['country']),
        #             'region': encode_cat_feature(interaction['region'], encoders_dict['region']),
        'city': encode_cat_feature(interaction['loc_id'], encoders_dict['city']),
        'os': encode_cat_feature(interaction['os_code'], encoders_dict['os']),
        #             'device': encode_cat_feature(interaction['deviceType'], encoders_dict['device']),
        #             'referrer_class': encode_cat_feature(interaction['referrerHostClass'], encoders_dict['referrer_class']),
    }


def split_sessions(group, encoders_dict, MAX_SESSION_IDLE_TIME_MS,
                   first_timestamp_ts):  # ,MAX_SESSION_IDLE_TIME_MS,first_timestamp_ts
    user, interactions = group
    # Ensuring items are sorted by time
    interactions_sorted_by_time = sorted(interactions, key=lambda x: x['time'])
    # Transforming interactions
    interactions_transformed = list(
        map(lambda interaction: transform_interaction(interaction, encoders_dict), interactions_sorted_by_time))

    sessions = []
    session = []
    first_timestamp = interactions_transformed[0]['timestamp']
    last_timestamp = first_timestamp
    for interaction in interactions_transformed:

        delta_ms = (interaction['timestamp'] - last_timestamp)
        interaction['_elapsed_ms_since_last_click'] = delta_ms

        if delta_ms <= MAX_SESSION_IDLE_TIME_MS:
            # Ignoring repeated items in session
            if len(list(filter(lambda x: x['article_id'] == interaction['article_id'], session))) == 0:
                session.append(interaction)
        else:
            # If session have at least 2 clicks (minimum for next click predicition)
            if len(session) >= 2:
                session_row = close_session(session, first_timestamp_ts)
                sessions.append(session_row)
            session = [interaction]

        last_timestamp = interaction['timestamp']

    if len(session) >= 2:
        session_row = close_session(session, first_timestamp_ts)
        sessions.append(session_row)

    # if len(sessions) > 1:
    #    raise Exception('USER with more than one session: {}'.format(user))

    return list(zip(map(lambda x: x['session_id'], sessions),
                    sessions))


def serialize(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)  # , protocol=pickle.HIGHEST_PROTOCOL)


# Read pickle file:

def loadPickle_2(name_file):
    # for reading also binary mode is important
    file = open(name_file, 'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file


def load_cafebiz_article_from_acr(cafebiz_article_from_acr_path):
    df = pd.read_csv(cafebiz_article_from_acr_path)
    return  df


def load_cafebiz_article_original_from_acr(input_articles_csv_path_original):
    df = pd.read_csv(input_articles_csv_path_original)
    return df


def handle_database_news_load_from_acr(input_articles_csv_path_original, spark):
    df = pd.read_csv(get_all_file(input_articles_csv_path_original)[0])

    # mySchema = StructType([StructField("category0", IntegerType(), True) \
    #                           , StructField("content", StringType(), True) \
    #                           , StructField("id", StringType(), True) \
    #                           , StructField("created_at_ts", IntegerType(), True) \
    #                           , StructField("teaser", StringType(), True) \
    #                           , StructField("domain", StringType(), True) \
    #                           , StructField("keywords", StringType(), True) \
    #                           , StructField("title", StringType(), True) \
    #                           , StructField("url", StringType(), True) \
    #                           , StructField("persons", StringType(), True) \
    #                           , StructField("locations", StringType(), True) \
    #                           , StructField("text_highlights", StringType(), True)])

    mySchema = StructType([StructField("id", LongType(), True) \
                              , StructField("content", StringType(), True) \
                              , StructField("created_at_ts", IntegerType(), True) \
                              , StructField("teaser", StringType(), True) \
                              , StructField("domain", StringType(), True) \
                              , StructField("keywords", StringType(), True) \
                              , StructField("title", StringType(), True) \
                              , StructField("url", StringType(), True) \
                              , StructField("category0", IntegerType(), True) \
                              , StructField("persons", StringType(), True) \
                              , StructField("locations", StringType(), True) \
                              , StructField("text_highlights", StringType(), True)])

    dbnews = spark.createDataFrame(df, schema=mySchema)
    dbnews = dbnews.select(regexp_replace(col("url"), "https://", "") \
                           .alias("full_url"), "id", "content", "created_at_ts", "teaser", "domain", "keywords",
                           "title", \
                           "url", "category0", "persons", "locations", "text_highlights")
    dbnews = dbnews.select(regexp_replace(col("full_url"), "http://", "") \
                           .alias("full_url"), "id", "content", "created_at_ts", "teaser", "domain", "keywords",
                           "title", \
                           "url", "category0", "persons", "locations", "text_highlights")

    return dbnews


def hard_code_log(spark, input_path_data_log_click):
    df = spark.read.parquet("hdfs://10.5.37.76:8020/Data/Logging/pvTosFull/pc/2019-05-01")
    df = df.filter(col("domain").isin(['cafebiz.vn']))
    df = df.withColumn('full_url', concat(col("domain"), col("path")))
    df = df.filter("guid != '-1'")

    df = df.withColumn("time", F.unix_timestamp(col("dt"), 'yyyy-MM-dd HH:mm:ss'))

    return df


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def get_list_hdfs_date(start_date, end_date, path_hdfs):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    list_path = []
    for single_date in daterange(start_date, end_date):
        path = path_hdfs + single_date.strftime("%Y-%m-%d") + '/'
        if os.system("hadoop fs -test -e %s" % path) == 0:
            list_path.append(path)
    return list_path


def handle_database_log_mysql(num_hour_trainning, domain, spark, mysql_host, mysql_user, mysql_passwd, mysql_database, mysql_table, current_time):
    from datetime import timedelta
    # import datetime
    date_time_obj = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    date_start = str(date_time_obj - timedelta(seconds=60*60*num_hour_trainning))
    date_end = current_time


    sql_query = "select * from "+mysql_table+" where domain = '"+domain+"' and create_time between '"+ date_start +"' and '"+date_end +"' ORDER BY create_time asc"
    print(sql_query)
    df_news = read_dbnews_mysql(mysql_host, mysql_user, mysql_passwd, mysql_database, sql_query)
    df_news = df_news.rename(columns={"id": "id", "create_time": "dt", "loc_id": "loc_id", "os_code": "os_code", "top": "top", \
                            "guid": "guid", "news_id": "news_id", "domain": "domain", "path": "path"})
    df_news = df_news.drop(columns='id')
    df_news['news_id'] = df_news['news_id'].astype('int64')
    # sua ten truong
    mySchema = StructType([StructField("dt", StringType(), True) \
                              , StructField("loc_id", IntegerType(), True) \
                              , StructField("os_code", IntegerType(), True) \
                              , StructField("top", IntegerType(), True) \
                              , StructField("guid", StringType(), True) \
                              , StructField("news_id", LongType(), True) \
                              , StructField("domain", StringType(), True) \
                              , StructField("path", StringType(), True) ])

    df = spark.createDataFrame(df_news, schema=mySchema)
    df = df.filter(col("domain").isin([domain]))
    df = df.withColumn('full_url', concat(col("domain"), col("path")))
    df = df.filter("guid != '-1'")

    df = df.withColumn("time", F.unix_timestamp(col("dt"), 'yyyy-MM-dd HH:mm:ss'))


    return df


def get_date_dt_data_log():
    from datetime import datetime
    now = datetime.now()
    datea = now.strftime("%Y-%m-%d %H:%M:%S")
    return datea

def deserialize(filename):
    #with open(filename, 'rb') as handle:
    with tf.gfile.Open(filename, 'rb') as handle:
        return pickle.load(handle)


def get_encoder_for_values_second(values, nar_encode_value, PAD_TOKEN='<PAD>', UNFREQ_TOKEN='<UNF>'):

    encoder_keys = [PAD_TOKEN, UNFREQ_TOKEN]
    encoder_values = [0,1]
    for i in range(0, len(values)):
        if values[i] in nar_encode_value.keys():
            encoder_keys.append(values[i])
            encoder_values.append(nar_encode_value[values[i]])
        else:
            encoder_keys.append(values[i])
            encoder_values.append(list(nar_encode_value.values())[1])
    encoder_dict = dict(zip(encoder_keys, encoder_values))
    return encoder_dict


def get_categ_features_encoder_dict_second_time(counts_df, nar_encode_value):
    freq_values = counts_df[counts_df['count'] >= 100][counts_df.columns[0]].values.tolist()
    encoder_dict = get_encoder_for_values_second(freq_values,nar_encode_value )
    return encoder_dict


def main_nar_preprocess_1():
    spark  = spark_inital()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    # get paramater
    parameter = load_json_config("./parameter.json")
    list_args = parameter["nar_preprocess_1"]

    DATA_DIR = parameter["DATA_DIR"]
    input_path_data_log_click = list_args["input_path_data_log_click"]
    date_start = list_args["date_start"]
    date_end = list_args["date_end"] 
    input_path_proprcessed_cafebiz_articale_csv_from_acr = DATA_DIR + list_args["input_path_proprcessed_cafebiz_articale_csv_from_acr"]
    nar_encoders_cafebiz = DATA_DIR + list_args["nar_encoders_cafebiz"]
    input_articles_csv_path_original = DATA_DIR + list_args["input_articles_csv_path_original"]
    mysql_host = list_args["mysql_host"]
    mysql_user = list_args["mysql_user"]
    mysql_passwd = list_args["mysql_passwd"]
    mysql_database = list_args["mysql_database"]
    mysql_table =  list_args["mysql_table"]
    domain = list_args["domain"]
    num_hour_trainning = list_args["n_hour_train_continue"]

    list_args2 = parameter["nar_preprocess_2"]
    spark_pre_json_path = DATA_DIR + list_args2['input_sessions_json_folder_path']
    session_tfrecord_path  = DATA_DIR + list_args2['output_sessions_tfrecords_path']

    # Delete folder before run nar
    if path.exists(spark_pre_json_path):
        import shutil
        shutil.rmtree(spark_pre_json_path)

    # delete_all_file_in_path(split_string_train_path( session_tfrecord_path))




    print("STARTNIG NAR PREPROCESSING ....")
    # database news
    print(input_articles_csv_path_original)
    df_news = handle_database_news_load_from_acr(input_articles_csv_path_original, spark)
    current_time = get_date_dt_data_log()
    flag = 0
    if path.exists(DATA_DIR+"/sessions_tfrecords_by_hour/"):
        # second time
        # read date now , 1 hour before
        df_log = handle_database_log_mysql(num_hour_trainning, domain, spark, mysql_host,mysql_user,mysql_passwd, mysql_database,mysql_table, current_time)
        flag = 2  # run second times

    else:
        # first time
        # read log from data_start to date_end
        df_log = handle_database_log(domain, spark, input_path_data_log_click, date_start, date_end)
        flag = 1 # first time

    # join database news and database log
    df = df_log.join(df_news, 'full_url', 'inner').drop(df_news.domain)
    # print(df.printSchema())

    print("<=== STARTING NAR PREPROCESSING 1 ===>")

    articles_original_df = load_cafebiz_article_from_acr(get_all_file(input_path_proprcessed_cafebiz_articale_csv_from_acr)[0])

    # code runable
    valid_articles_urls_to_ids_dict = dict(
        articles_original_df[['url', 'id_encoded']].apply(lambda x: (x['url'], x['id_encoded']), axis=1).values)

    valid_articles_urls_to_ids_dict_broadcast = spark.sparkContext.broadcast(valid_articles_urls_to_ids_dict)

    ### Loading user interactions
    ### TEST data_test
    # df = spark.read.parquet("file:///data1/ngocvb/Ngoc_COV/chameleon/nar_preprocess/nar_data/data_test/")

    #     df = spark.read.parquet("file:///data/tungtv/jupytercode/data-log-news-parquet-thang45/")
    # df_news = handle_database_news_load_from_acr(
    #     "/data/tungtv/Code/dataset/dataset_cafebiz_acr_nar_1/original_cafebiz_articles_csv/cafebiz_articles_original.csv",
    #     spark)
    # df_log = handle_database_log(spark, "hdfs://10.5.37.76:8020/Data/Logging/pvTosFull/pc/", "2019-07-01", "2019-07-05")
    # df = df_log.join(df_news, 'full_url', 'inner').drop(df_news.domain)

    interactions_df = df.select("full_url", "dt", "os_code", "loc_id", "path" \
                                , "guid", "category0", "id", "content", "created_at_ts" \
                                , "teaser", "title", "keywords", "time", "url", "top")
    # .alias("full_url"), "id", "content", "created_at_ts", "teaser", "domain", "keywords",
    # "title", \
    # "url", "category0", "persons", "locations", "text_highlights")
    # tructField("dt", StringType(), True) \
    #     , StructField("loc_id", IntegerType(), True) \
    #     , StructField("os_code", IntegerType(), True) \
    #     , StructField("top", IntegerType(), True) \
    #     , StructField("guid", StringType(), True) \
    #     , StructField("news_id", LongType(), True) \
    #     , StructField("domain", StringType(), True) \
    #     , StructField("path", StringType(), True)])

    interactions_df = interactions_df.withColumn("id", interactions_df["id"].cast(LongType()))
    interactions_df = interactions_df.withColumn("created_at_ts", interactions_df["created_at_ts"].cast(LongType()))

    get_article_id_encoded_from_url_udf = F.udf(
        lambda url: get_article_id_encoded_from_url(url, valid_articles_urls_to_ids_dict_broadcast),
        pyspark.sql.types.IntegerType())

    # Filtering only interactions whose url/id is available in the articles table
    # tungtv
    interactions_article_id_encoded_df = interactions_df.withColumn('article_id', get_article_id_encoded_from_url_udf(
        interactions_df['url']))
    #     interactions_article_id_encoded_df = interactions_df.withColumn('article_id', interactions_df['id'])

    interactions_filtered_df = interactions_article_id_encoded_df.filter(
        interactions_article_id_encoded_df['article_id'].isNull() == False)

    # print(interactions_filtered_df.printSchema())
    # print(interactions_filtered_df.select("article_id"))
    first_timestamp_ts = interactions_filtered_df.select('time').agg(F.min('time')).collect()[0][0] * 1000

    # Analyzing elapsed time since publishing
    interactions_filtered_df = add_column_dataframe(interactions_filtered_df, "publish_ts", F.to_timestamp(
        interactions_filtered_df.created_at_ts.cast(dataType=t.TimestampType())))

    interactions_filtered_df = add_column_dataframe(interactions_filtered_df, "publish_ts",
                                                    F.unix_timestamp(col("publish_ts"), 'yyyy-MM-dd HH:mm:ss'))

    get_timestamp_from_date_str_udf = F.udf(get_timestamp_from_date_str, pyspark.sql.types.IntegerType())

    interactions_filtered_with_publish_ts_df = add_column_dataframe(interactions_filtered_df,
                                                                    "elapsed_min_since_published", \
                                                                    ((F.col('time') - F.col('publish_ts')) / 60).cast(
                                                                        pyspark.sql.types.IntegerType()))

    interactions_filtered_with_publish_ts_df.approxQuantile("elapsed_min_since_published",
                                                            [0.10, 0.25, 0.50, 0.75, 0.90], 0.01)

    elapsed_min_since_published_df = interactions_filtered_with_publish_ts_df.select(
        'elapsed_min_since_published').toPandas()

    """PAD_TOKEN = '<PAD>'
    UNFREQ_TOKEN = '<UNF>'"""

    # Analyzing clicks by article distribution
    ## Processing categorical features
    if flag == 1:
        cities_df = get_categ_features_counts_dataframe(interactions_filtered_df, 'loc_id')
        cities_encoder_dict = get_categ_features_encoder_dict(cities_df)

        os_df = get_categ_features_counts_dataframe(interactions_filtered_df, 'os_code')
        os_encoder_dict = get_categ_features_encoder_dict(os_df)
    else:
        # map value from nar encode city
        # read nar_encode_pickle
        nar_encoder_dict =  deserialize(get_all_file(nar_encoders_cafebiz)[0])
        cities_df = get_categ_features_counts_dataframe(interactions_filtered_df, 'loc_id')
        cities_encoder_dict = get_categ_features_encoder_dict_second_time(cities_df,nar_encoder_dict['os'] )

        os_df = get_categ_features_counts_dataframe(interactions_filtered_df, 'os_code')
        os_encoder_dict = get_categ_features_encoder_dict_second_time(os_df,nar_encoder_dict['city'])

    encoders_dict = {
        'city': cities_encoder_dict,
        #     'region': regions_encoder_dict,
        #     'country': countries_encoder_dict,
        'os': os_encoder_dict,
        #     'device': devices_encoder_dict,
        #     'referrer_class': referrer_class_encoder_dict
    }

    # Processing numeric features

    active_time_quantiles = interactions_filtered_df.approxQuantile("top", [0.10, 0.25, 0.50, 0.75, 0.90], 0.01)

    active_time_stats_df = interactions_filtered_df.describe('top').toPandas()

    active_time_mean = float(active_time_stats_df[active_time_stats_df['summary'] == 'mean']['top'].values[0])
    active_time_stddev = float(active_time_stats_df[active_time_stats_df['summary'] == 'stddev']['top'].values[0])

    interactions_filtered_df = interactions_filtered_df.withColumnRenamed("guid", "user_id")
    interactions_filtered_df = interactions_filtered_df.orderBy("time")

    ### Splitting sessions
    MAX_SESSION_IDLE_TIME_MS = 30 * 60 * 1000  # 30 min
    # test_df = interactions_filtered_df.limit(1000).rdd.map(lambda x: (x['user_id'], x))
    # print(test_df.take(10))
    sessions_rdd = interactions_filtered_df.rdd.map(lambda x: (x['user_id'], x)).groupByKey() \
        .flatMap(lambda row: split_sessions(row, encoders_dict, MAX_SESSION_IDLE_TIME_MS, first_timestamp_ts)) \
        .sortByKey() \
        .map(lambda x: x[1])

    #### Exporting sessions to JSON lines
    sessions_sdf = sessions_rdd.toDF()

    sessions_sdf.write.partitionBy("session_hour").json(os.path.join(DATA_DIR, "sessions_processed_by_spark/"))

    if path.exists(nar_encoders_cafebiz):
        pass
    else:
        os.makedirs(nar_encoders_cafebiz)

    if flag == 1: # first time
        NAR_ENCODERS_PATH = 'nar_encoders_cafebiz.pickle'
        serialize(nar_encoders_cafebiz + NAR_ENCODERS_PATH, encoders_dict)

    print(" <=== END NAR PREPROCESSING 1 ===>")

if __name__ == '__main__':
    main_nar_preprocess_1()

