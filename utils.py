import os
import shutil


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