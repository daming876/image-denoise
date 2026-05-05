import shutil
import os
import random
from tqdm import tqdm

img_path = './exploration_database_and_code/pristine_images'   #项目本地文件路径
copy_to_path = './exploration_database_and_code/train'

for i in range(3000):
    i = i+1
    img = str("%05d" % i)+'.bmp'
    # 图片复制到另一个文件夹，把img_path这个文件夹下的img拷贝到copy里下夹文件夹下
    shutil.copy(os.path.join(img_path, img), os.path.join(copy_to_path, img))
    #os.remove(os.path.join(img_path, img))#并删除原有文件


