import os
import pickle
import shutil
import logging
from tensorboardX import SummaryWriter


def check_and_create_path(path):
    if os.path.isdir(path):
        print("Experiment in {} already present".format(path))
        inp = input("Press e to exit, r to replace it: ")
        if inp == "e":
            exit()
        elif inp == "r":
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print("Input not understood")
            exit()
    else:
        os.makedirs(path)

def save_object(obj, path):
    with open('{}/obj_dump.pkl'.format(path), 'wb') as fileobj:
        pickle.dump(obj, fileobj)


class Logger():
    def __init__(self, log_dir) -> None:
        log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
                     "%(filename)s::%(lineno)d::%(message)s"
        logging.basicConfig(filename="{log_path}/log_console.log".format(
                                                     log_path=log_dir),
                            level='DEBUG', format=log_format)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.log_dir = log_dir

    def init_tb(self):
        tb_path = self.log_dir + "/tensorboard"
        # if not os.path.exists(tb_path) or not os.path.isdir(tb_path):
        os.makedirs(tb_path)
        self.writer = SummaryWriter(tb_path)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)
