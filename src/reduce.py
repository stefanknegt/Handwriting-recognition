import os, shutil


def create_reduced(folder):
    orig = '../data/Train/annotated_crops/'+folder
    red = '../data/Train/annotated_crops/red_'+folder
    if not os.path.exists(red):
        os.makedirs(red)
    for dir in os.listdir(orig):
        path_or = os.path.join(orig, dir)
        path_red = os.path.join(red, dir)
        if len(os.listdir(path_or))>4:
            print(dir)
            if not os.path.exists(path_red):
                os.makedirs(path_red)
            for filename in os.listdir(path_or):
                shutil.copy(os.path.join(path_or, filename), path_red)
       
if __name__ == "__main__":
    create_reduced('128_extended_bin')