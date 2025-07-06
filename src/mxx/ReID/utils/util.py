import importlib
import random

def get_utils(id_dataset):
    current_package = __name__.rsplit('.', 1)[0]
    # print(current_package)
    # exit()
    module_name = f".version.{id_dataset}"
    module = importlib.import_module(module_name, package=current_package)
    return module