from flask import request, Blueprint
import app.modulehelloworld.service.hello_world as hw
from flask_api import status
# Create blueprint
from nar_module.nar.nar_trainer_cafebiz_full import NAR_Model_Predict
from pick_singleton.pick_singleton import ACR_Pickle_Singleton
import json

simple_page = Blueprint('simple_page', __name__,
                        template_folder='templates')
def load_json_config(path_config_file):
    with open(path_config_file) as config_file:
        data = json.load(config_file)
    return data
# Sample routing
@simple_page.route('/hello', methods=['POST', 'GET'])
def hello():
    return 'Hello, World'


@simple_page.route('/helloworld', methods=['POST', 'GET'])
def hello_message():
    name = request.values.get("name")
    return hw.hello_world(name)

@simple_page.route('/loadacr', methods=['POST', 'GET'])
def load_acr():
    try:
        parameter = load_json_config("./parameter.json")
        acr_path = parameter["DATA_DIR"] + "/pickles/acr_articles_metadata_embeddings_predict/acr_articles_metadata_embeddings_predict.pickle"

        acr_label_encoders_predict, articles_metadata_df_predict, content_article_embeddings_matrix_predict = \
            hw.load_acr_module_resources(acr_path)
        reverse_acr_article_id = {v:k  for k,v in acr_label_encoders_predict['article_id'].items()}
        acr_pickle_singleton = ACR_Pickle_Singleton()
        acr_pickle_singleton.getUpdateInstance(acr_label_encoders_predict, reverse_acr_article_id,articles_metadata_df_predict, content_article_embeddings_matrix_predict )

        # test acr changed
        acr = ACR_Pickle_Singleton().getInstance()
        print(len(acr.acr_label_encoders["article_id"].keys()))
        print("Updated ACR SingleToon")
        nar = NAR_Model_Predict()
        nar.getUpdateInstance()
        print("Load acr by url Done")

        # # Call any predict to load nar before
        # import requests
        # # resp = requests.get('http://0.0.0.0:8082/recommend?news_id=0&user_id=0')
        # resp = requests.get('http://0.0.0.0:8082/recommend?news_id=20190918203115156&user_id=0')
        # print(resp.status_code)
        # if resp.status_code == 200:
        #     print("Predict any case success")
        # else:
        #     print("Predict any case fail")
        return "Loaded ACR Pickle", status.HTTP_200_OK
    except Exception as ex:
        print("Exception Load ACR")
        print(ex)
        raise


@simple_page.route('/recommend', methods=['POST', 'GET'])
def recommend():
    return 'Recommend'