import schedule
import time
import sys
import json
sys.path.append("/data1/tungtv/code/chameleon/newsrecomdeepneural")
print(sys.path)

from nar_module.nar.preprocessing.nar_preprocess_cafebiz_1 import main_nar_preprocess_1
from nar_module.nar.preprocessing.nar_preprocess_cafebiz_2 import main_nar_preprocess_2
from nar_module.nar.run_nar_train import main_nar_train


def run_acr_schedule():
    main_nar_preprocess_1()
    main_nar_preprocess_2()
    main_nar_train()

def load_json_config(path_config_file):
    with open(path_config_file) as config_file:
        data = json.load(config_file)
    return data

parameter = load_json_config("./parameter.json")
list_args = parameter["nar_preprocess_1"]
num_hour_trainning = list_args["n_hour_train_continue"]

def main():
    schedule.every(1).hours.do(run_acr_schedule)
    while True:
        schedule.run_pending()
        time.sleep(1)

    # run_acr_schedule()

if __name__ == '__main__':
    main()