from acr_module.acr.acr_module_service import load_json_config
# from nar_module.nar.nar_trainer_cafebiz_full import NAR_Model_Predict
from nar_module.nar.nar_model import get_list_id
from nar_module.nar.nar_utils import load_nar_module_preprocessing_resources
from nar_module.nar.utils import deserialize
from nar_module.nar.benchmarks import SequentialRulesRecommender
class Singleton(type):
    """
    An metaclass for singleton purpose. Every singleton class should inherit from this class by 'metaclass=Singleton'.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



class ACR_Pickle_Singleton(object,metaclass=Singleton):
   __instance = None
   def __init__(self):
       parameter = load_json_config("./parameter.json")
       list_args = parameter["acr_preprocess"]
       self.acr_path = parameter["DATA_DIR"] + "/pickles/acr_articles_metadata_embeddings/acr_articles_metadata_embeddings.pickle"
       self.model_nar_dir = parameter["DATA_DIR"] + parameter["model_dir_nar"]
       print("Singleton ACR Init")
       (acr_label_encoders, articles_metadata_df, content_article_embeddings) = \
           deserialize(self.acr_path)
       self.acr_label_encoders = acr_label_encoders
       self.articles_metadata_df =articles_metadata_df
       # self.reverse_acr_article_id = {}
       self.reverse_acr_article_id = {v: k for k, v in acr_label_encoders['article_id'].items()}
       self.content_article_embeddings_matrix = content_article_embeddings

       # get list_id
       list_id = get_list_id()
       encoded_list_id=[]

       for id in list_id:
           if (int(id) in acr_label_encoders['article_id']):
               encoded_list_id.append(self.get_article_id_encoded(int(id)))
       list_id_week_encoded = list(articles_metadata_df['article_id'])[-600:]
       encoded_list_id = list_id_week_encoded + encoded_list_id

       list_id_week = list(acr_label_encoders['article_id'].keys())[-600:]
       self.list_id = list(list_id) + list_id_week
       self.encoded_list_id = encoded_list_id
       # print("Loading ACR singleton")
       ACR_Pickle_Singleton.__instance = self
       print("Done SingleTon Init Done")

   @staticmethod
   def getInstance():
       if ACR_Pickle_Singleton.__instance == None:
           # print("ACR is none")
           ACR_Pickle_Singleton()
       return ACR_Pickle_Singleton.__instance

   def getUpdateInstance(self, acr_label_encoders,reverse_acr_article_id, articles_metadata_df, content_article_embeddings_matrix):
       self.acr_label_encoders = acr_label_encoders
       self.reverse_acr_article_id = reverse_acr_article_id
       self.articles_metadata_df = articles_metadata_df
       self.content_article_embeddings_matrix = content_article_embeddings_matrix
       # print("ACR singleton update")
       ACR_Pickle_Singleton.__instance = self
       print("Into ACR Update Instance")
       # NAR_Model_Predict.getUpdateInstance()
       return  self.acr_label_encoders ,  self.articles_metadata_df, self.content_article_embeddings_matrix


   def getUpdateInstance_byFlask(self):
       print("Into ACR Update Instance by Flask")
       parameter = load_json_config("./parameter.json")
       list_args = parameter["acr_preprocess"]
    #    acr_path = parameter["DATA_DIR"] + "/pickles/acr_articles_metadata_embeddings_predict/acr_articles_metadata_embeddings_predict.pickle"
    #    acr_path = parameter["DATA_DIR"] + "/pickles/acr_articles_metadata_embeddings/acr_articles_metadata_embeddings.pickle"
       self.model_nar_dir = parameter["DATA_DIR"] + parameter["model_dir_nar"]

       (acr_label_encoders, articles_metadata_df, content_article_embeddings) = \
           deserialize(self.acr_path)
       self.acr_label_encoders = acr_label_encoders
       self.articles_metadata_df = articles_metadata_df
       # self.reverse_acr_article_id = {}
       self.reverse_acr_article_id = {v: k for k, v in acr_label_encoders['article_id'].items()}
       self.content_article_embeddings_matrix = content_article_embeddings

       # get list_id
       list_id = get_list_id()
       encoded_list_id = []

       for id in list_id:
           if (int(id) in acr_label_encoders['article_id']):
               encoded_list_id.append(self.get_article_id_encoded(int(id)))
       list_id_week_encoded = list(articles_metadata_df['article_id'])[-600:]
       encoded_list_id = list_id_week_encoded + encoded_list_id
       list_id_week = list(acr_label_encoders['article_id'].keys())[-600:]
       self.list_id = list(list_id) + list_id_week
       self.encoded_list_id = encoded_list_id

       # print("Loading ACR singleton")
       ACR_Pickle_Singleton.__instance = self
       print("Done Update SingleTon Flask Init Done")

   def get_article_id(self, article_id_encoded):

       try:
            return str(self.reverse_acr_article_id[article_id_encoded])
       except Exception as ex:
           return self.reverse_acr_article_id['article_id'][0]

   def get_article_id_encoded(self,article_id):
       try:
            return self.acr_label_encoders['article_id'][article_id]
       except Exception as ex:
           return self.acr_label_encoders['article_id']['<PAD>']

class NAR_Pickle_Singleton(object,metaclass=Singleton):
   __instance = None
   def __init__(self):
       parameter = load_json_config("./parameter.json")
       list_args = parameter["acr_preprocess"]
       nar_path = parameter["DATA_DIR"] + "/pickles/nar_preprocessing_resources/nar_preprocessing_resources.pickle"
       self=  load_nar_module_preprocessing_resources(nar_path)
       # print("Loading NAR singleton")
       NAR_Pickle_Singleton.__instance = self

   @staticmethod
   def getInstance():
       if NAR_Pickle_Singleton.__instance == None:
           # print("NAR singleton is none")
           NAR_Pickle_Singleton()
       return NAR_Pickle_Singleton.__instance

   def getUpdaetInstance(self):
       print("Into update nar encoder singleton")
       parameter = load_json_config("./parameter.json")
       list_args = parameter["acr_preprocess"]
       nar_path = parameter["DATA_DIR"] + "/pickles/nar_preprocessing_resources/nar_preprocessing_resources.pickle"
       self = load_nar_module_preprocessing_resources(nar_path)
       # print("Loading NAR singleton")
       NAR_Pickle_Singleton.__instance = self

from nar_module.nar.datasets import prepare_dataset_iterator
from nar_module.nar.utils import resolve_files, chunks
import tensorflow as tf

class SRModel_Singleton(object,metaclass=Singleton):
    __instance = None
    def __init__(self,training_hour=7*24,batch_size=64,truncate_session_length=20):
        self.params = {'max_clicks_dist': 10, #Max number of clicks to walk back in the session from the currently viewed item. (Default value: 10) 
                        'dist_between_clicks_decay': 'div' #Decay function for distance between two items clicks within a session (linear, same, div, log, qudratic). (Default value: div) 
                    }
        self.training_hour = training_hour
        self.batch_size = batch_size
        self.truncate_session_length = truncate_session_length
        self.clf = SequentialRulesRecommender(None, self.params, None)
        self.predictions = {}
        SRModel_Singleton.__instance = self

    @staticmethod
    def getInstance():
        if SRModel_Singleton.__instance == None:
            # print("NAR singleton is none")
            SRModel_Singleton()
        return SRModel_Singleton.__instance
    
    def update_rule(self):
        clf = SequentialRulesRecommender(None, self.params, None)
        clf.rules['test'] = 2
        nar_label_encoders = NAR_Pickle_Singleton.getInstance()
        self.train(clf,nar_label_encoders)

    def train(self, clf,nar_label_encoders):
        from nar_module.nar.nar_trainer_cafebiz_full import get_session_features_config
        session_features_config = get_session_features_config(nar_label_encoders)
        train_data = self.get_training_files(self.training_hour)
        # print(train_data)
        it = prepare_dataset_iterator(train_data, session_features_config, 
                                                                        batch_size=self.batch_size,
                                                                        truncate_session_length=self.truncate_session_length)
        count = 0
        with tf.Session() as sess:
            while True:
                try:
                    data_it = sess.run(it)
                    #convert encoded id to id
                    acr_pickle = ACR_Pickle_Singleton.getInstance()
                    # print("BEFORE")
                    # print("CLICKED")
                    # print(data_it[0]['item_clicked'])
                    # print("LABEL")
                    # print(data_it[1]['label_next_item'])
                    # data_it[0]['item_clicked'].astype(str)
                    # data_it[1]['label_next_item'].astype(str)
                    self.convert_encoded_ids(data_it[0]['item_clicked'])
                    self.convert_encoded_ids(data_it[1]['label_next_item'])
                    count +=1
                    clf.train(data_it[0]['user_id'],data_it[0]['session_id'],
                    data_it[0]['item_clicked'],data_it[1]['label_next_item'])
                except tf.errors.OutOfRangeError:
                    break   
        print("Total training sample: "+ str(count*self.batch_size))
        # print("-----------------")
        # print(clf.rules)
        # print("int 20190607183749984")
        # print(clf.rules[20190607183749984])
        # print("str 20190607183749984")
        # print(clf.rules["20190607183749984"])
        # print(self.clf.rules[5108])
        self.clf = clf
    def convert_encoded_ids(self,clickeds_all_items):
        acr_pickle = ACR_Pickle_Singleton.getInstance()
        for i,clicked_items in enumerate(clickeds_all_items):
            for j,item in enumerate(clicked_items):
                if item != 0:
                    clickeds_all_items[i][j] = acr_pickle.get_article_id(item)
    def get_training_files(self, training_hour):
        parameter  = load_json_config("./parameter.json")
        training_dir = parameter["DATA_DIR"]+ parameter["nar_preprocess_2"]["output_sessions_tfrecords_path"]
        train_files = resolve_files(training_dir)[-training_hour:]
        # print("TrainFile")
        # print(train_files)
        return list(chunks(train_files, training_hour))

    def predict(self, item_id, valid_items, topk=100,topk_per_item=10):
        return self.clf.predict_topk(item_id, topk, topk_per_item, valid_items)
    
    # def pre_calculate_result(news_id_list):


def main():
    nar_label_encoders = NAR_Pickle_Singleton.getInstance()
    srclassifier = SRModel_Singleton()
    sr = SequentialRulesRecommender(None, srclassifier.params, None)
    sr.rules['test'] = 1
    srclassifier.train(sr,nar_label_encoders)
    print(srclassifier.clf.rules['test'])
    # srclassifier.update_rule()
    # print(srclassifier.clf.rules['test'])
    # print("------------------")
    # print(srclassifier.predict(3648,100,10,None))
    # print(len(set(srclassifier.predict(3648,100,10,None))))

if __name__ == '__main__':  
    main()