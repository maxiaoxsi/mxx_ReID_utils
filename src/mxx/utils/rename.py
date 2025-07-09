import os

def rename_dir(dir_base, name, name_new):
    for root, dirs, files in os.walk(dir_base):
        for dir in dirs:
            if dir == name:
                os.raneme(os.path.join(root, name), os.path.join(root, name_new))

if __name__ == '__main__':
    dir_base = '/machangxiao/datasets/ReID_smplx/MARS-v160809/bbox_train'
    

