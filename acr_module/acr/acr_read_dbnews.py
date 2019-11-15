import pandas as pd

import os
# import findspark
# findspark.init()
# from pyspark import SparkContext, SparkConf,SQLContext
# from pyspark.sql import SparkSession, Row
#
# from pyspark.sql.types import StringType, StructType, StructField
# from pyspark.sql.types import ArrayType
# from pyspark.sql.functions import udf


from bs4 import BeautifulSoup
from underthesea import ner
from underthesea import word_tokenize
import mysql.connector

# def spark_initial():
#     '''
#     config spark to can run spark in your machine
#     :return:
#     '''
#
#     spark = SparkSession.builder.master('local[*]').appName('myAppName') \
#         .config("spark.local.dir", "/data1/tungtv/tmp/").getOrCreate()
#
#     sc = spark.sparkContext
#     sqlContext = SQLContext(sc)
#
#     return spark, sqlContext


def read_dbnews_mysql(v_host, v_user, v_pass, v_db, sql_query):
    mydb = mysql.connector.connect(
        host=v_host,
        user=v_user,
        passwd=v_pass,
        database=v_db)

    df = pd.read_sql(sql_query, con=mydb)
    return df


# remove tag web in content
def remove_tag_web(string):
    soup = BeautifulSoup(string, "html.parser")
    str_resutl = soup.get_text()
    bad_chars = ['"',';', ':', '!', "*","(", ")", "{", "}", "/", "|", "-", "_",\
                "=","+", "&", "^", "%", "$", "#", "@", "!","<", ">", "~",\
                "()"]
    str_resutl = ''.join(i for i in str_resutl if not i in bad_chars)
    return str(str_resutl)

# find persion , location entities
def find_per(text):
    per = []
    for x in ner(text):
        if "PER" in x[3]:
            per.append(x[0])
    per_str = ','.join(per)
    return per_str

# find persion , location entities
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