import argparse
from mxx.utils.rename import rename_dir
from mxx.utils.process_dir import process_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_base", type=str)
    parser.add_argument("--type_process", type=str)
    parser.add_argument("--name", type=str, default="smplx")
    parser.add_argument("--name_new", type=str, default="pred")
    args = parser.parse_args()
    dir_base = args.dir_base
    type_process = args.type_process
    if type_process == "rename":
        args = (args.name, args.name_new)
    elif type_process == "delete":
        args = (args.name, )
    else:
        raise Exception("wrong!")
    process_dir(
        dir_base=dir_base, 
        type_process=type_process, 
        args=args
    )
    