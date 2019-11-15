import schedule
import time
import sys
import acr_module.acr.run_acr as run_acr
import nar_module.nar.run_nar as run_nar
from threading import Thread
import threading
import time
from multiprocessing import Pool
from multiprocessing import Process, Value
import multiprocessing as multip
from multiprocessing import Process, Manager



def main():
    # print(nar.getNarDict)
    pool = Pool(processes=3)
    parsed = pool.apply_async(run_acr.main)
    pattern = pool.apply_async(run_nar.main)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()