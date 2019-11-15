import argparse
import pandas as pd
import datetime
import re
import os.path
from os import path
import json
import mysql.connector
import pickle
import operator
from bs4 import BeautifulSoup
from pyspark import SparkContext, SparkConf,SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, LongType

import warnings



warnings.filterwarnings('ignore')

from underthesea import ner
from datetime import timedelta
from datetime import timedelta



import pyspark
from pyspark.sql.functions import col
# findspark.init()
from pyspark.sql.functions import *

import pyspark.sql.types as T
import pyspark.sql.functions as F

from pyspark.sql import types as t

from underthesea import word_tokenize


from scipy import stats
from pyspark.sql.functions import to_timestamp

import schedule
from dateutil.parser import parse
from joblib import Parallel, delayed
from sklearn.utils import class_weight
from collections import defaultdict, Counter
# from pyspark.sql.functions import udf
# from pyspark.sql.types import StringType, StructType, StructField
# from pyspark.sql.types import ArrayType
import _thread
import time

import nltk


#import tensorflow as tf
import sys



# sys.path.append("/data1/tungtv/code/chameleon/newsrecomdeepneural")

from acr_module.acr.tf_records_management import export_dataframe_to_tf_records, make_sequential_feature
from acr_module.acr.utils import serialize, chunks, get_categ_encoder_from_values, encode_categ_feature, resolve_files, deserialize, serialize, log_elapsed_time
from acr_module.acr.preprocessing.tokenization import tokenize_articles, nan_to_str, convert_tokens_to_int, get_words_freq, \
    convert_tokens_to_int_second_time
from acr_module.acr.preprocessing.word_embeddings import load_word_embeddings_vietnamese, load_word_embeddings, \
    process_word_embedding_for_corpus_vocab, save_word_vocab_embeddings, \
    process_word_embedding_for_corpus_vocab_second_time
from acr_module.acr.acr_model import ACR_Model
from sklearn.preprocessing import StandardScaler

from acr_module.acr.preprocessing.acr_preprocess_cafebiz import (process_cat_features, save_article_cat_encoders,
    make_sequence_example, load_input_csv, custome_df, get_tkn_fn, load_acr_preprocessing_label_encode,
    process_cat_features_second_time, get_date_time_current, load_acr_preprocessing_word_embedding)


def create_args_parser_acr():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_word_embeddings_path', default='',
        help='Input path of the word2vec embeddings model (word2vec).')

    parser.add_argument(
        '--max_words_length', type=int, default=1000,
        help='Maximum tokens length of text.')

    parser.add_argument(
        '--output_tf_records_path', default='',
        help='Output path for generated TFRecords with news content.')

    parser.add_argument(
        '--output_word_vocab_embeddings_path', default='',
        help='Output path for a pickle with words vocabulary and corresponding word embeddings.')

    parser.add_argument(
        '--output_label_encoders', default='',
        help='Output path for a pickle with label encoders for categorical features.')

    parser.add_argument(
        '--output_articles_csv_path_preprocessed', default='',
        help='Output path for a CSV file with articles contents.')

    parser.add_argument(
        '--articles_by_tfrecord', type=int, default=1000,
        help='Number of articles to be exported in each TFRecords file')

    parser.add_argument(
        '--vocab_most_freq_words', type=int, default=100000,
        help='Most frequent words to keep in vocab')

    parser.add_argument(
        '--mysql_host', type=str,
        help='Mysql Host.')

    parser.add_argument(
        '--mysql_user', type=str,
        help='Mysql User.')

    parser.add_argument(
        '--mysql_passwd', type=str,
        help='Mysql Password.')

    parser.add_argument(
        '--mysql_database', type=str,
        help='Mysql Database.')

    parser.add_argument(
        '--path_pickle', type=str,
        help='Path of pickle file.')

    parser.add_argument(
        '--path_tf_record', type=str,
        help='Path of tf record.')
    return parser

def get_all_file(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.pickle' in file:
                files.append(os.path.join(r, file))
            if '.tfrecord' in file:
                files.append(os.path.join(r, file))
            if '.csv' in file:
                    files.append(os.path.join(r, file))
    return files

def split_string(string):
    return string.split('/')[-1]

def extract_date(string):
    return (re.findall('\d+', string))

def split_date( s ):
    try:
        return int((s.split("_")[-1]).split(".")[0])
    except ValueError:
        return ""

def get_file_max_date(list_name):
    date = []
    name_file = ""
    for name in list_name:
        date_2 = split_date(split_string(name))
        date.append(date_2)
        if date_2 == max(date):
            name_file = name
    return name_file

def get_listmax_date(list_name):
    try:
        date = []
        list_file = []
        for name in list_name:
            date.append(split_date(split_string(name)))

        for name in list_name:
            if split_date(split_string(name)) == max(date):
                list_file.append(name)
        return sorted(list_file)
    except ValueError:
        return []

def load_model_word2vec(path_model_word2vec):
    w2v_model = load_word_embeddings_vietnamese(path_model_word2vec, binary=False)
    return w2v_model

def write_date(date):
    data = {}
    data['date'] = date
    with open('./acr_module/config/date.json', 'w') as outfile:
        json.dump(data, outfile)

def load_date(path_file):
    with open(path_file) as config_file:
        data = json.load(config_file)
    return data

def handle_database_news(doamin, mysql_host,mysql_user, mysql_passwd,mysql_database, mysql_table, v_date_start, v_date_end):


    date_json = load_date("./acr_module/config/date.json")
    # date_json = load_date("/home/tungtv/Documents/Code/News/newsrecomdeepneural/acr_module/config/date.json")
    start_date_from_json = date_json["date"]
    if start_date_from_json == "":
        date_start = v_date_start
    else:
        print("date_start = date in json")
        date_time_obj =  datetime.datetime.strptime(start_date_from_json, '%Y-%m-%d %H:%M:%S')
        date_start = str(date_time_obj + timedelta(seconds=1))

    if v_date_end < date_start:
        print("v_date_end < date_start")
        date_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_end = v_date_end

    print("Reading database ... ")
    sql_query = "select * from "+mysql_table+"  where sourceNews = '"+doamin+"' and publishDate BETWEEN '"+date_start+"' AND '"+date_end+"' ORDER BY publishDate asc"
    print(sql_query)

    df = read_dbnews_mysql( mysql_host,  mysql_user,  mysql_passwd, mysql_database, sql_query)

    if df.empty :
        pass
    else:
        write_date(str(df['publishDate'].iloc[-1]))

    return df
    # df = pd.read_json("/home/tungtv/Documents/Code/News/dataset/dataset_cafebiz_mini_45/data-thang-4-5-10r-new.json")
    # df = df[:10]
    # print(df.columns)

    # df = df[df["sourceNews"] == "CafeBiz"]
    # df = df[df["is_deleted"] == 0]
    # list_cul = ['catId', 'content', 'newsId', 'publishDate', 'sapo', 'sourceNews', 'tags', 'title', 'url']
    # df = df[list_cul]
    # df = df.dropna()
    #
    # # find entites (person, location)
    # print("Find entities ...")
    # df['tags'] = df['tags'].apply(lambda x: pre_keyword(x))
    # df['content'] = df['content'].apply(lambda x: remove_tag_web(x))
    # df['persons'] = df['content'].apply(lambda x: find_per(x))
    # df['locations'] = df['content'].apply(lambda x: find_loc(x))
    #
    # df['tags'].str.lower()
    # df['content'].str.lower()
    # df['persons'].str.lower()
    # df['locations'].str.lower()



    # spark, sqlContext = spark_initial()
    # df_spark = spark.createDataFrame(df)
    # # remove tag web by spark
    # remove_tag = udf(lambda content: remove_tag_web(content), StringType())
    # df_spark = df_spark.withColumn("content", remove_tag(df_spark.content))
    # # find persons by spark
    # find_per_udf = udf(lambda content: find_per(content), StringType())
    # df_spark = df_spark.withColumn("persons", find_per_udf(df_spark.content))
    # # find location by spark
    # find_loc_udf = udf(lambda content: find_loc(content), StringType())
    # df_spark = df_spark.withColumn("locations", find_loc_udf(df_spark.content))
    # df = df_spark.toPandas()


    # df = df.rename(columns={"catId": "category0", "newsId": "id", "publishDate": "created_at_ts", \
    #                         "sapo": "teaser", "title": "title", "tags": "keywords", "sourceNews": "domain"})
    # df['text_highlights'] = df['title'] + "|" + df['teaser'] + "|" + df['content']
    # df['created_at_ts'] = pd.to_datetime(df['created_at_ts']).astype(int) // 10 ** 9

    # list_col = ["id", "content", "created_at_ts", "teaser", "domain", "keywords", "title", "url", "category0",
    #             "persons", "locations"]
    #
    # df = df[list_col]
    # return df

class Singleton:
   __instance = None
   @staticmethod
   def getInstance(path_word2vec):
      """ Static access method. """
      if Singleton.__instance == None:
         Singleton(path_word2vec)
      return Singleton.__instance


   def __init__(self, path_word2vec):
      """ Virtually private constructor. """
      if Singleton.__instance != None:
          raise Exception("This class is a singleton!")
      else:
          print("Load model word2vec")
          self = load_word_embeddings_vietnamese(path_word2vec, binary=False)
          Singleton.__instance = self

def load_json_config(path_config_file):
    with open(path_config_file) as config_file:
        data = json.load(config_file)
    return data

def read_dbnews_mysql(v_host, v_user, v_pass, v_db, sql_query):
    mydb = mysql.connector.connect(
        host=v_host,
        user=v_user,
        passwd=v_pass,
        database=v_db,
        port=3306)

    df = pd.read_sql(sql_query, con=mydb)
    return df

def remove_tag_web(string):
    soup = BeautifulSoup(string, "html.parser")
    str_resutl = soup.get_text()
    bad_chars = ['"',';', ':', '!', "*","(", ")", "{", "}", "/", "|", "-", "_",\
                "=","+", "&", "^", "%", "$", "#", "@", "!","<", ">", "~",\
                "()"]
    str_resutl = ''.join(i for i in str_resutl if not i in bad_chars)
    return str(str_resutl)

def find_per(text):
    per = []
    for x in ner(text):
        if "PER" in x[3]:
            per.append(x[0])
    per_str = ','.join(per)
    return per_str

def find_loc(text):
    loc = []
    for x in ner(text):
         if "LOC" in x[3]:
            loc.append(x[0])
    loc_str = ','.join(loc)
    return loc_str

def pre_keyword(text):
    lista = text.split(';')
    list_kw = ','.join(lista)
    return list_kw

def write_dict_newsid_encode(dictinary_news_id_encode):
    pickle_out = open("./acr_module/config/dict_news_id_encode.pickle", "wb")
    pickle.dump(dictinary_news_id_encode, pickle_out)
    pickle_out.close()

def load_dict_news_id_encode():
    pickle_in = open("./acr_module/config/dict_news_id_encode.pickle", "rb")
    dict1 = pickle.load(pickle_in)
    return dict1

def Merge(dict1, dict2):
    return(dict2.update(dict1))

def append_dict_newsid_encode(dict_id_new):
    if os.path.exists('./acr_module/config/dict_news_id_encode.pickle'):
        pickle_in = open("./acr_module/config/dict_news_id_encode.pickle", "rb")
        dict_id_old = pickle.load(pickle_in)

        # dict_id_new = acr_label_encoders['article_id']
        Merge(dict_id_old, dict_id_new)

        # acr_label_encoders['article_id'] = dict_id_new
        dict_id_new = sorted(dict_id_new.items(), key=operator.itemgetter(1))
        dict_id_new = dict(dict_id_new)

def write_articale_csv_original(news_df, output_articles_csv_path_original):
    # load df originla cu
    # append them df moi
    # luu xuong
    if len(os.listdir(output_articles_csv_path_original)) == 0:
        news_df.to_csv(output_articles_csv_path_original + "cafebiz_articles_original.csv", index=False)
    else:
        news_df_old = pd.read_csv(get_all_file(output_articles_csv_path_original)[0])
        frames = [news_df_old, news_df]

        news_df_new= pd.concat(frames,ignore_index=True)

        news_df_new.to_csv(output_articles_csv_path_original+"cafebiz_articles_original.csv", index=False)

def xuly(df):

    df = df[df["sourceNews"] == "CafeBiz"]
    # df = df[df["is_deleted"] == 0]
    list_cul = ['catId', 'content', 'newsId', 'publishDate', 'sapo', 'sourceNews', 'tags', 'title', 'url']
    df = df[list_cul]
    df = df.dropna()

    # find entites (person, location)
    print("Find entities ...")
    df['tags'] = df['tags'].apply(lambda x: pre_keyword(x))
    df['content'] = df['content'].apply(lambda x: remove_tag_web(x))
    df['persons'] = df['content'].apply(lambda x: find_per(x))
    df['locations'] = df['content'].apply(lambda x: find_loc(x))

    df['tags'].str.lower()
    df['content'].str.lower()
    df['persons'].str.lower()
    df['locations'].str.lower()


    df = df.rename(columns={"catId": "category0", "newsId": "id", "publishDate": "created_at_ts","sapo": "teaser", "title": "title", "tags": "keywords", "sourceNews": "domain"})
    df['text_highlights'] = df['title'] + "|" + df['teaser'] + "|" + df['content']
    df['created_at_ts'] = pd.to_datetime(df['created_at_ts']).astype(int) // 10 ** 9

    list_col = ["id", "content", "created_at_ts", "teaser", "domain", "keywords", "title", "url", "category0",
                "persons", "locations", "text_highlights"]

    df = df[list_col]
    print("Xu Ly Xong")
    return df

def spark_inital():
    # spark = SparkSession.builder.master('local[30]').appName('ACR Preprocess').config("spark.hadoop.yarn.resourcemanager.address", "172.18.5.69:8032").config("spark.local.dir", "/data1/tungtv/tmp/").getOrCreate()
    # spark = SparkSession.builder.master('local[10]').appName('ACR Preprocess').config("spark.hadoop.yarn.resourcemanager.address", "10.5.36.95:8032").config("spark.local.dir","/data1/tungtv/tmp/").getOrCreate()
    # spark = SparkSession.builder.master('local[*]').appName('myAppName').getOrCreate()
    spark = SparkSession.builder.master('local[10]').appName('ACR Preprocess').config("spark.local.dir","/data1/tungtv/tmp/").config("spark.driver.memory", "12g").getOrCreate()

    # data: 69
    # data1: 95
    return spark

def remove_tag_web(string):
    soup = BeautifulSoup(string, "html.parser")
    str_resutl = soup.get_text()

    bad_chars = ['"', ';', ':', '!', "*", "(", ")", "{", "}", "/", "|", "-", "_", \
                     "=", "+", "&", "^", "%", "$", "#", "@", "!", "<", ">", "~", \
                     "()"],

    str_resutl = ''.join(i for i in str_resutl if not i in bad_chars)

    return str(str_resutl)

# find person  location
def find_per_loc(text):

    # text = """ "Nỗi ám ảnh khách hàng""Trong kinh doanh, các lãnh đạo của Amazon luôn lấy khách hàng làm gốc, không ngừng tìm kiếm và nõ lực gìn giữ lòng tin của khách hàng. Dù có quan tâm đến đối thủ nhưng họ thường bị ám ảnh bởi khách hàng nhiều hơn"", John Rossman- cựu giám đốc dịch vụ doanh nghiệp của Amazon.com nhận xét trong cuốn sách viết về phương thức kinh doanh của gã khổng lồ thương mại điện tử này.  Theo Rossman, sự ám ảnh về khách hàng của Jeff Bezos là một thứ gì đó vượt xa nỗi ám ảnh đơn thuần- nó là hội chứng tâm lý khiến Jeff có thể buông những lời chỉ trích cay độc hay đưa ra nhiều nhận định khắt khe đối với cộng sự tại Amazon khi họ không đáp ứng được các tiêu chuẩn của Jeff về dịch vụ khách hàng. Điều này bắt nguồn từ khả năng đặc biệt của Jeff trong việc đặt bản thân vào vị trí của khách hàng, từ đó suy ra những nhu cầu và mong muốn của họ, để sau đó phát triển một hệ thống có thể đáp ứng các nhu cầu và mong muốn đó tốt hơn bất kỳ ai.  ""Cách tiếp cận kinh doanh này chính là điểm cốt lõi trong tài năng thiên bẩm của Jeff"", Rossman nhận định. Cụ thể là rất lâu trước khi truyền thông xã hội tạo ra cuộc cách mạng rộng rãi trong thế giới bán lẻ, khi những mạng lưới thông tin minh bạch được hình thành, kết nối các công ty với khách hàng hiện tại, khách hàng tiềm năng và cả những kẻ gièm pha; rất lâu trước khi những công ty như Zappos.com bắt đầu lấy dịch vụ khách hàng làm nền tảng cho mô hình kinh doanh; cũng như rất lâu trước khi Jeff nhận ra tầm nhìn toàn diện của bản thân đối với Amazon.com, ông đã quán triệt sâu sắc trong nội bộ công ty về 2 chân lý trong lĩnh vực dịch vụ khách:  Một là Khi một công ty khiến cho một khách hàng không hài lòng, người đó sẽ không chỉ nói điều đó với một, hai hay ba người khác, mà sẽ nói với rất, rất nhiều người.  Hai là Dịch vụ khách hàng tốt nhất là không có dịch vụ nào hết- bởi trải nghiệm tốt nhất có được khi khách hàng không bao giờ phải yêu cầu một sự hỗ trợ nào cả.  Hạn chế tối đa sự tham gia của con người  Tất nhiên, việc xây dựng một mô hình kinh doanh thực tế không yêu cầu bất kỳ dịch vụ khách hàng nào cũng viễn tưởng như việc chế tạo động cơ vĩnh cửu. Nhưng ngay từ giai đoạn đầu của cuộc cách mạng Internet, Jeff đã nhận thấy mô hình bán lẻ trực tuyến có thể mở đường cho nhiều điều khả thi. Từ lâu, ông cũng đã nhận ra mối đe dọa lớn nhất đối với trải nghiệm khách hàng là khi con người tham gia vào và làm cho mọi thứ rối tung lên. Do đó, ông đưa đến kết luận, chìa khóa để tạo ra trải nghiệm dễ chịu và suôn sẽ nhất cho khách hàng là hạn chế tối đa sự tham gia của con người thông qua quá trình đổi mới và công nghệ.  Tất nhiên Amazon vẫn cần con người. Và Jeff Bezos là người có kỹ năng trong việc tuyển dụng, đánh giá và giữ chân những nhân tài hàng đầu thế giới. Nhưng mục tiêu của Amazon luôn là giảm thiểu thời gian và công sức mà con người phải tiêu tốn vào những tương tác dịch vụ thông thường, giải phóng sức lao động để họ có thể sáng tạo những cách thức mới làm hài lòng khách hàng.  Quan điểm kinh doanh của Jeff đã dẫn tới một số chiến thuật khá khác thường. Trở lại thời điểm những năm 1990, Amazon.com cố tình làm cho việc tìm kiếm só điện thoại dịch vụ chăm sóc khách hàng trở nên khó khăn. Điều này khiến nhiều người nghi ngại và cho rằng động thái đó phản ánh sự không tôn trọng khách hàng. Nhưng rồi họ nhanh chóng nhận ra rằng, những kỹ sư của Jeff đã tạo ra một công nghệ đột phá giúp họ giải quyết những yêu cầu dịch vụ gần như ngay lập tức mà không cần đến sự can thiệp của con người. Sau tất cả, 98% các câu hỏi của người mua dành cho một nhà bán lẻ như Amazon đều quy về ""Đồ của tôi đang ở đâu?"" Do đó, một công cụ theo dõi trực tuyến ra đời, giúp khách hàng theo dõi việc vận chuyển hàng từ kho tới tận cửa nhà, bỏ qua yêu cầu phải có một trung tâm liên lạc lớn, cồng kềnh và tốn kém.  Jeff tin rằng mọi người không thích nói chuyện với các nhân viên chăm sóc khách hàng, và ông đã đúng. Ông chỉ cung cấp dữ liệu, công cụ và hướng dẫn để người mua có thể trả lời những câu hỏi của chính họ.  Giờ đây khách hàng đều chờ đợi và yêu cầu ""công nghệ chăm sóc khách hàng tự phục vụ"" do Bill Price và David Jaffe dưa ra trong cuốn sách The best service is no service vào năm 2008: Trải nghiệm của khách hàng càng trơn tru, khách hàng càng trugn thành và chi phí vận hành càng thấp (bao gồm cả chi phí marketing và quảng cáo).  Hai tác giả này lý giải: ""Amazon đã giảm được 90% lượng liên lạc mỗi đơn hàng (CPO- Contacts per order), nghĩa là họ có thể tăng số lượng đơn hàng (và doanh thu) lên gấp 9 lần trong khi vẫn duy trì mức chi phí chăm sóc khách hàng như cũ (bao gồm chi phí nhân viên và các chi phí vận hành liên quan). Đây là nhân tố chủ chốt góp phần vào khả năng sinh lời của công ty bắt đầu từ năm 2002"".  (*) Nội dung tham khảo cuốn sách Phương thức Amazon- Tác giả John Rossman, Vũ Khánh Thịnh dịch.Quyền lực đáng sợ của Amazon: Các công ty 'lèo tèo' làm ngơ cũng không được, cạnh tranh cũng chẳng xong, đành phải hợp tác" """
    per = []
    loc = []
    for x in ner(text):
        if "PER" in x[3]:
            per.append(x[0])
        if "LOC" in x[3]:
            loc.append(x[0])
    if len(loc) == 0:
        loc.append("0")
    if len(per) == 0:
        per.append("0")
    return Row("persons", "locations")(",".join(per), ",".join(loc))

def handle_database_news_by_spark(df_news, spark):
    # list_cul = ['catId', 'content', 'email', 'newsId', 'publishDate', 'sapo', 'sourceNews', 'tags', \
    #             'title', 'url']
    # df_news = df_news[df_news["sourceNews"] == "CafeBiz"]
    # df_news = df_news.dropna()
    #
    # # remove news deleted
    # df_news = df_news[df_news["is_deleted"] == 0]
    #
    # df_news = df_news[list_cul]

    df = spark.createDataFrame(df_news)

    remove_tag = udf(lambda content: remove_tag_web(content), StringType())
    df = df.withColumn("content", remove_tag(df.content))

    schema = StructType([
        StructField("persons", StringType(), False),
        StructField("locations", StringType(), False)])

    find_udf = udf(find_per_loc, schema)

    df = df.withColumn("Output", find_udf(df.content))

    df = df.withColumnRenamed("catId", "category0")
    df = df.withColumnRenamed("newsId", "id")
    df = df.withColumnRenamed("publishDate", "created_at_ts")
    df = df.withColumnRenamed("sapo", "teaser")
    df = df.withColumnRenamed("title", "title")
    df = df.withColumnRenamed("tags", "keywords")
    df = df.withColumnRenamed("sourceNews", "domain")

    df = df.drop("email")

    df = df.select("id", "content", "created_at_ts", "teaser", "domain", "keywords", \
                   "title", "url", "category0", "Output.*")

    # df = df.na.fill(0)

    # to pandas
    pdd = df.toPandas()

    pdd['text_highlights'] = pdd['title'] + "|" + pdd['teaser'] + "|" + pdd['content']

    pdd['created_at_ts'] = pd.to_datetime(pdd['created_at_ts']).astype(int) // 10 ** 9
    pdd['keywords'] = pdd['keywords'].apply(lambda x: pre_keyword(x))
    # pdd = pdd.fillna(0,inplace=True)

    # pdd.to_parquet("/data/tungtv/Code/dataset/dataset_cafebiz_acr_nar_1/data-thang78-final.parquet",
    #                compression="snappy", engine='fastparquet')
    return pdd


def remove_duplicate_newsid(newsdf, acr_label_encoders):
    #     acr_label_encoders, articles_metadata_df, content_article_embeddings_matrix = deserialize(acr_content_path)
    list_id = set(acr_label_encoders["article_id"].keys())

    list_id_news_df = set(newsdf['id'])

    #     list_id_intersec =  list(set(newsdf['id']).intersection(set(acr_label_encoders["article_id"].keys())))
    list_id_intersec = list(list_id.intersection(list_id_news_df))

    #     for i in range(0, len(list_id_intersec)):
    #         newsdf = newsdf[newsdf['id'] != list_inter[i]]

    newsdf = newsdf[~newsdf['id'].isin(list_id_intersec)]
    return newsdf

def main_acr_preprocess():
# def main():

    parameter = load_json_config("./parameter.json")
    list_args = parameter["acr_preprocess"]
    DATA_DIR = parameter["DATA_DIR"]
    path_pickle = DATA_DIR+list_args["path_pickle"]
    path_tf_record  = DATA_DIR+list_args["path_tf_record"]
    input_word_embeddings_path = DATA_DIR+list_args["input_word_embeddings_path"]
    vocab_most_freq_words = list_args["vocab_most_freq_words"]
    max_words_length = list_args["max_words_length"]
    output_word_vocab_embeddings_path = DATA_DIR+list_args["output_word_vocab_embeddings_path"]
    output_label_encoders = DATA_DIR+list_args["output_label_encoders"]
    output_tf_records_path = DATA_DIR+list_args["output_tf_records_path"]
    output_articles_csv_path_preprocessed = DATA_DIR+list_args["output_articles_csv_path_preprocessed"]
    output_articles_csv_path_original = DATA_DIR+list_args["output_articles_csv_path_original"]
    articles_by_tfrecord = list_args["articles_by_tfrecord"]
    mysql_host = list_args["mysql_host"]
    mysql_user = list_args["mysql_user"]
    mysql_passwd = list_args["mysql_passwd"]
    mysql_database = list_args["mysql_database"]
    mysql_table = list_args["mysql_table"]
    domain = list_args["domain"]


    print("<=== STARTING ARC PREPROCESS ===>")
    spark  = spark_inital()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)


    # ACR PREPROCESS
    if path.exists(output_label_encoders):
        pass
    else:
        os.makedirs(output_label_encoders)
        os.makedirs(output_word_vocab_embeddings_path)
        os.makedirs(output_articles_csv_path_preprocessed)
        os.makedirs(output_articles_csv_path_original)

    if path.exists( path_tf_record):
        pass
    else:
        os.makedirs( path_tf_record)

    list_args_2 = parameter["acr_training"]
    acr_path = DATA_DIR + list_args_2["output_acr_metadata_embeddings_path"]
    acr_label_encoders, articles_metadata_df, content_article_embeddings = deserialize(get_all_file(acr_path)[0])
    isEmpty = 0
    # DATABASE NEWS FOMR MYSQL
    news_df_from_mysql = handle_database_news(domain, mysql_host,  mysql_user,  mysql_passwd, mysql_database,mysql_table, list_args['date_start'], list_args["date_end"])

    # filter
    list_cul = ['catId', 'content', 'email', 'newsId', 'publishDate', 'sapo', 'sourceNews', 'tags','title', 'url']
    news_df_from_mysql = news_df_from_mysql[news_df_from_mysql["sourceNews"] == "CafeBiz"]
    news_df_from_mysql = news_df_from_mysql.dropna()

    # remove news deleted
    news_df_from_mysql = news_df_from_mysql[news_df_from_mysql["is_deleted"] == 0]

    news_df_from_mysql = news_df_from_mysql[list_cul]

    if news_df_from_mysql.empty:  # if empty
        isEmpty = 1
        return isEmpty

    news_df_handle_by_spark = handle_database_news_by_spark(news_df_from_mysql, spark)

    news_df = remove_duplicate_newsid(news_df_handle_by_spark, acr_label_encoders)

    if news_df.empty:  # if empty
        isEmpty = 1
        return isEmpty

    # fill 0 for NaN values entities (person, location)
    # news_df.fillna("0", inplace=True)
    news_df = custome_df(news_df)
    print("Saving news articles csv original CSV to ")
    write_articale_csv_original(news_df, output_articles_csv_path_original)



    if len(os.listdir( path_pickle)) == 0: # empty first time
        print("File Chua Da Ton Tai")
        print('Encoding categorical features')
        cat_features_encoders, labels_class_weights = process_cat_features(news_df)

        # write_dict_newsid_encode(cat_features_encoders["article_id"])

        print('Exporting LabelEncoders of categorical features: {}'.format( output_label_encoders))
        save_article_cat_encoders( output_label_encoders +"acr_label_encoders.pickle", cat_features_encoders,
                                         labels_class_weights)

        print("Saving news articles CSV to {}".format( output_articles_csv_path_preprocessed))
        # news_df.to_csv( output_articles_csv_path_preprocessed +"cafebiz_articles.csv", index=False)
        news_df.to_csv(output_articles_csv_path_preprocessed + "cafebiz_articles.csv", index=False)

        print('Tokenizing articles...')
        tokenized_articles = tokenize_articles(news_df['text_highlights'].values,
                                                   tokenization_fn=get_tkn_fn( max_words_length))

        print('Computing word frequencies...')
        words_freq = get_words_freq(tokenized_articles)

        print("Loading word2vec model and extracting words of this corpus' vocabulary...")
        w2v_model = Singleton.getInstance(input_word_embeddings_path)
        word_vocab, word_embeddings_matrix = process_word_embedding_for_corpus_vocab(w2v_model,
                                                                                     words_freq,
                                                                                      vocab_most_freq_words)

        print('Saving word embeddings and vocab.: {}'.format( output_word_vocab_embeddings_path))
        save_word_vocab_embeddings( output_word_vocab_embeddings_path +"acr_word_vocab_embeddings.pickle",
                                       word_vocab, word_embeddings_matrix)

        print('Converting tokens to int numbers (according to the vocab.)...')
        texts_int, texts_lengths = convert_tokens_to_int(tokenized_articles, word_vocab)
        news_df['text_length'] = texts_lengths
        news_df['text_int'] = texts_int

        data_to_export_df = news_df[['id', 'url',  # For debug
                                         'id_encoded',
                                         'category0_encoded',
                                         # 'category1_encoded',
                                         'keywords_encoded',
                                         # 'author_encoded',
                                         # 'concepts_encoded',
                                         # 'entities_encoded',
                                         'locations_encoded',
                                         'persons_encoded',
                                         'created_at_ts',
                                         'text_length',
                                         'text_int']]

        print("Category 0:", news_df["category0_encoded"].unique())
        for k, v in labels_class_weights.items():
            print("Label class weight shape:", k, ":", v.shape)
        print('Exporting tokenized articles to TFRecords: {}'.format(path_tf_record))
        export_dataframe_to_tf_records(data_to_export_df,
                                       make_sequence_example,
                                       output_path= output_tf_records_path,
                                       examples_by_file= articles_by_tfrecord)

    else: # not empty run more than one time
        print("File Da Ton Tai")
        print("Database have new article")
        print("Call singelton ACR content: ")
         
        #Load ACR content
        # #1
        # from pick_singleton.pick_singleton import ACR_Pickle_Singleton
        # acr_content = ACR_Pickle_Singleton.getInstance()
        # acr_label_encoders = acr_content.acr_label_encoders

        #2

        # dict_news_id_encode = load_dict_news_id_encode()
        # word_vocab, word_embeddings_matrix = load_acr_preprocessing_word_embedding(get_file_max_date(get_all_file( output_word_vocab_embeddings_path)))

        print('Encoding categorical features')
        cat_features_encoders, labels_class_weights = process_cat_features_second_time(news_df, acr_label_encoders)

        # append_dict_newsid_encode(cat_features_encoders["article_id"])
        # write_dict_newsid_encode(cat_features_encoders["article_id"])

        # print('Exporting LabelEncoders of categorical features: {}'.format( output_label_encoders))
        # save_article_cat_encoders( output_label_encoders +"acr_label_encoders.pickle", cat_features_encoders,
        #                               labels_class_weights)

        print("Saving news articles CSV to {}".format( output_articles_csv_path_preprocessed ))
        path_csv = get_all_file(output_articles_csv_path_preprocessed)[0]
        df = pd.read_csv(path_csv)
        frames = [df, news_df]

        result = pd.concat(frames, ignore_index=True)
        result.to_csv( output_articles_csv_path_preprocessed +"cafebiz_articles.csv", index=False)


        print('Tokenizing articles...')
        tokenized_articles = tokenize_articles(news_df['text_highlights'].values,
                                                   tokenization_fn=get_tkn_fn( max_words_length))

        print('Computing word frequencies...')
        words_freq = get_words_freq(tokenized_articles)

        print("Loading word2vec model and extracting words of this corpus' vocabulary...")
        w2v_model = Singleton.getInstance(input_word_embeddings_path)
        word_vocab, word_embeddings_matrix = process_word_embedding_for_corpus_vocab(w2v_model,
                                                                                         words_freq,
                                                                                          vocab_most_freq_words )

        print('Saving word embeddings and vocab.: {}'.format( output_word_vocab_embeddings_path))
        # save_word_vocab_embeddings( output_word_vocab_embeddings_path +"acr_word_vocab_embeddings.pickle",
        #                                word_vocab, word_embeddings_matrix)

        print('Converting tokens to int numbers (according to the vocab.)...')
        texts_int, texts_lengths = convert_tokens_to_int_second_time(tokenized_articles, word_vocab)
        news_df['text_length'] = texts_lengths
        news_df['text_int'] = texts_int

        data_to_export_df = news_df[['id', 'url',  # For debug
                                         'id_encoded',
                                         'category0_encoded',
                                         # 'category1_encoded',
                                         'keywords_encoded',
                                         # 'author_encoded',
                                         # 'concepts_encoded',
                                         # 'entities_encoded',
                                         'locations_encoded',
                                         'persons_encoded',
                                         'created_at_ts',
                                         'text_length',
                                         'text_int']]

        print("Category 0:", news_df["category0_encoded"].unique())
        for k, v in labels_class_weights.items():
            print("Label class weight shape:", k, ":", v.shape)
        print('Exporting tokenized articles to TFRecords: {}'.format( path_tf_record))
        print("len data_to_export_df : {}".format(len(data_to_export_df)))
        export_dataframe_to_tf_records(data_to_export_df,
                                           make_sequence_example,
                                           output_path= output_tf_records_path,
                                           examples_by_file= articles_by_tfrecord)

    print("<=== END ARC PREPROCESS ===>")
    return isEmpty


if __name__ == '__main__':
    main_acr_preprocess()

