# Import flask and template operators
from flask import Flask, render_template
import atexit
import json
import glob
import os
import shutil

from nar_module.nar.nar_trainer_cafebiz_full import NAR_Model_Predict
from pick_singleton.pick_singleton import ACR_Pickle_Singleton, NAR_Pickle_Singleton, SRModel_Singleton
from apscheduler.schedulers.background import BackgroundScheduler

# Define the WSGI application object

# Configurations
# app.config.from_object('config')

def delete_all_file_in_path(path):
    files = glob.glob(path+'*')
    for f in files:
        os.remove(f)
def load_json_config(path_config_file):
    with open(path_config_file) as config_file:
        data = json.load(config_file)
    return data

def update_model():
    acr = ACR_Pickle_Singleton()
    acr.getUpdateInstance_byFlask()
    nar_encoder = NAR_Pickle_Singleton()
    nar_encoder.getUpdaetInstance()
    # nar = NAR_Model_Predict()
    # nar.getUpdateInstance()
    classifier = SRModel_Singleton()
    classifier.update_rule()

def remove_all_dir_export():
    parameter = load_json_config("./parameter.json")
    export_dir = parameter['DATA_DIR']+'/model_nar/exported/'

    # shutil.rmtree(export_dir)
    #
    # import os
    # var1 = 'mkdir' + export_dir
    # print(var1)
    # myCmd = var1
    # if os.system(myCmd) != 0:
    #     print("Xoa va tao  thanh cong")
    # else:
    #     print("Xoa va tao That bai")

    for root, dirs, files in os.walk(export_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

update_model()
#
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_model, trigger="interval",minutes=10)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())
atexit.register(remove_all_dir_export)

app = Flask(__name__)

# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    print(error)
    return render_template('404.html'), 404


# Import a module / component using its blueprint handler variable
from app.modulehelloworld.controller.hello_world import simple_page

#ngocvb
from app.module_recommend.controller.recommend import recommend_page
# Register blueprint(s)
app.register_blueprint(recommend_page)

# Register blueprint(s)
app.register_blueprint(simple_page)


