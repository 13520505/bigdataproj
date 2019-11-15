import subprocess
import sys
sys.path.append("/data/tungtv/Code/NewsRecomDeepLearning")

def main_nar_train():
    subprocess.call("./nar_module/scripts/run_nar_train_cafebiz.sh",  shell=True)

if __name__ == '__main__':
    main_nar_train()