import os
import shutil

def rename_dir(dir, root, args):
    (name, name_new) = args
    if dir == name:
        os.raneme(os.path.join(root, name), os.path.join(root, name_new))

def delete_dir(root, dir, args):
    (name,) = args
    if dir == name:
        try:
            shutil.rmtree(os.path.join(root, name))
            print(f'delete dir {os.path.join(root, name)}')
        except OSError as e:
            print(f"[process_dir] root: {root}, name:{name} delete dir error!")

def process_dir(dir_base, type_process, args):
    if type_process == "rename":
        method = rename_dir
    elif type_process == "delete":
        method = delete_dir
    for root, dirs, files in os.walk(dir_base):
        for dir in dirs:
            method(root, dir, args)

