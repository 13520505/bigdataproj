import subprocess
import schedule
import time
import os
import sys
# sys.path.append("/home/tungtv/Documents/Code/News/newsrecomdeepneural")
# print(sys.path)
from acr_module.acr.acr_module_service import main_acr_preprocess
from acr_module.acr.acr_trainer_cafebiz import main_acr_train



def run_acr_preprocess_train():
    isEmpty = main_acr_preprocess()
    if isEmpty == 1:
        return
    else:
        main_acr_train()

def main():
    # schedule.every(10).seconds.do(run_acr_preprocess_train)
    count = 0
    while True:
        # if count >= 1:
            # break
        # schedule.run_pending()
        # print("Running ACR ...")
        run_acr_preprocess_train()
        # break
        time.sleep(10)
        # count += 1
    print("RUN ACR DONE")

if __name__ == '__main__':
    main()