# from nar_module.nar.nar_trainer_cafebiz_full import predict
import nar_module.nar.nar_trainer_cafebiz_full as nar
import tensorflow as tf
import json
import time
#set param
from nar_module.nar.RecentlyPopularRecommender import RecentlyPopularRecommender
from pick_singleton.pick_singleton import ACR_Pickle_Singleton, SRModel_Singleton

def recommendSR(news_id, user_id):
    try:
        pickle = ACR_Pickle_Singleton().getInstance()
        SRModel = SRModel_Singleton().getInstance()
        id_encoded = pickle.get_article_id_encoded(int(news_id))
        predict1 = {}
        pred0 = {}
        pre_list = []

        guid = user_id
        if id_encoded==0:
            start_time1 = time.time()
            list_id_pop = RecentlyPopularRecommender().get_recent_popular_item_ids()[:100]
            for k in range(0,len(list_id_pop)):
                pred0['id'] = list_id_pop[k]
                pre_list.append(pred0.copy())
            print("--- %s seconds recommend pop news ---" % (time.time() - start_time1))
        else:
            start_time1 = time.time()
            list_predict = SRModel.predict(int(news_id), pickle.list_id)
            # print(list_predict)
            for article_id in list_predict:
                # encode_id -> news_id
                # article_id = pickle.get_article_id(encoded_id)
                pred0['id'] = str(article_id)
                # pred0['id'] = x['top_k_predictions'][-1][k].tolist()
                pre_list.append(pred0.copy())
            print("--- %s seconds recommend SR news ---" % (time.time() - start_time1))
        predict1['recommend'] = pre_list
        predict1['guid'] = str(guid)
        predict1['algid'] = 33
        pre_string1 = json.dumps(predict1, indent=4, separators=(',', ': '))
        return pre_string1
    except Exception as ex:
        print("Exception Recommend")
        print(ex)
        raise ex

def recommend(news_id, user_id):

    pickle = ACR_Pickle_Singleton().getInstance()
    id_encoded = pickle.get_article_id_encoded(int(news_id))
    # print(id_encoded)

    if (user_id is None or len(user_id) == 0) and (news_id is None or len(news_id) == 0) :
        return "Hello !!!"
    try:
        if (id_encoded == 0):
            start_time1 = time.time()
            predict1 = {}
            pred0 = {}
            pre_list = []

            guid = user_id
            list_id_pop = RecentlyPopularRecommender().get_recent_popular_item_ids()[:100]

            for k in range(0,len(list_id_pop)):
                pred0['id'] = list_id_pop[k]
                pre_list.append(pred0.copy())
            predict1['recommend'] = pre_list
            predict1['guid'] = str(guid)
            predict1['algid'] = 32
            pre_string1 = json.dumps(predict1, indent=4, separators=(',', ': '))
            print("--- %s seconds recommend pop news ---" % (time.time() - start_time1))
            return pre_string1
        else:
            # print("====================> INTO RECOMMEND")
            # print("Into preidct chamemleon")
            start_time = time.time()
            list_predict = nar.predict(news_id, user_id)
            # print("--- %s seconds list_predict ---" % (time.time() - start_time))
            # result = list(list_predict)
            #json_predict = []
            predict1 = {}
            # start_time_acr = time.time()
            pickle = ACR_Pickle_Singleton().getInstance()
            # print("--- %s seconds call acr singelton ---" % (time.time() - start_time_acr))
            # print(len(list_predict))

            pred0 = {}
            pre_list = []
            # start2 = time.time()
            encoded_top_k = list_predict['top_k_predictions'][0:,-1,:100]
            # top_k = encoded_top_k[0].tolist()
            for k in range(0,len(encoded_top_k[0])):
                # encode_id -> news_id
                article_id = pickle.get_article_id(encoded_top_k[0][k])
                pred0['id'] = article_id
                # pred0['id'] = x['top_k_predictions'][-1][k].tolist()
                pre_list.append(pred0.copy())
                # print("--------%s second decode newsid--------" %(time.time()-start2))
                #pre_string0 = json.dumps(pred, indent=4, sort_keys=True, separators=(',', ': '))
                #item_clicked = x['item_clicked'].tolist()
            # print("Loadd return list id success")
            # print("--------%s second decode newsid--------" %(time.time()-start2))
            guid = list_predict['user_id'][0].decode("utf-8")
            #session_id = int(x['session_id'])
            # json_predict.append(guid)
            # json_predict.append(session_id)
            predict1['recommend'] = pre_list
            predict1['guid'] = guid
            predict1['algid'] = 32
            # predict1['session_id'] = session_id
            #predict1['item_clicked']=item_clicked
            # print("--- %s seconds predict ---" % (time.time() - start_time))
            # values = ','.join(str(v) for v in list_predict)
            # json_predict = json.dumps(list_predict)
            # start_time_json = time.time()
            pre_string1 = json.dumps(predict1, indent=4,  separators=(',', ': '))
            # print("--- %s json dumps ---" % (time.time() - start_time_json))
            print("--- %s seconds recommend ---" % (time.time() - start_time))
            return pre_string1
    except Exception as ex:
        print("Exception Recommend")
        print(ex)
        raise ex


