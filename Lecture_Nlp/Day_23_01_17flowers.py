# Day_23_01_17flowers.py
import tensorflow as tf
import os
from PIL import Image       # Pillow


# 문제
# 17flowers_origin 폴더의 이미지를 (224, 224)로 줄여서 17flowers_224에 복사하세요

# 문제
# 17flowers 폴더를 읽어서 x, y 데이터를 반환하는 함수를 만드세요
# x: 4차원
# y: 1차원(1~80은 0번, 81~160은 1번, ...)


def resize_17flowers(src_folder, dst_folder, new_size):
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    for filename in sorted(os.listdir(src_folder)):
        if filename.startswith('.'):
            continue

        # print(filename)

        img_1 = Image.open(os.path.join(src_folder, filename))
        img_2 = img_1.resize([new_size, new_size])
        img_2.save(os.path.join(dst_folder, filename))


def get_xy(data_folder):
    pass


# resize_17flowers('17flowers_origin', '17flowers_56', new_size=56)
# resize_17flowers('17flowers_origin', '17flowers_112', new_size=112)
# resize_17flowers('17flowers_origin', '17flowers_224', new_size=224)

get_xy('17flowers_56')



