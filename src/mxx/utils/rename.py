import os
import shutil

def rename_dir(dir, root, args):
    (name, name_new) = args
    if dir == name:
        os.raneme(os.path.join(root, name), os.path.join(root, name_new))

def delete_dir(dir, root, args):
    (name,) = args
    if dir == name:
        os.defpath

def process_dir(dir_base, type_process, args):
    if type_process == "rename":
        method = rename_dir
    elif type_process == "delete":
        nethod = delete_dir
    for root, dirs, files in os.walk(dir_base):
        for dir in dirs:
            method(root, dir, args)

