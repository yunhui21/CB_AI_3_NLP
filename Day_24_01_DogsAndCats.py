# Day_24_01_DogsAndCats.py
# dogs_and_cats
import tensorflow as tf
import numpy as np
import shutil
import os
from PIL import Image

'''
train
small
    train
        cat
        dog
    validation
        cat
        dog
    test
        cat
        dog
'''

# 문제
# 위와 같은 형식의 폴더를 생성하는 함수를 만드세요.

# 문제
# 원본 train 폴더의 이미지를
# small/train 1000개, small/validation에 500개, small/test에 500개
# 파일 이름에 들어있는 규칙을 활용하세요.

def make_dataset_folders():
    def make_if_not(dst_folder):
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

    # make_if_not('DogsandCats/small/train')
    # make_if_not('DogsandCats/small/validation')
    # make_if_not('DogsandCats/small/test')
    #
    # make_if_not('DogsandCats/small/train/dogs')
    # make_if_not('DogsandCats/small/train/cata')
    # make_if_not('DogsandCats/small/validation/dogs')
    # make_if_not('DogsandCats/small/validation/cats')
    # make_if_not('DogsandCats/small/test/dogs')
    # make_if_not('DogsandCats/small/test/cats')

    def make_if_not_2(dst_folder):
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)


    make_if_not_2('DogsandCats/small/train/dogs')
    make_if_not_2('DogsandCats/small/train/cats')
    make_if_not_2('DogsandCats/small/validation/dogs')
    make_if_not_2('DogsandCats/small/validation/cats')
    make_if_not_2('DogsandCats/small/test/dogs')
    make_if_not_2('DogsandCats/small/test/cats')

# def make_small_datasets():
#     def copy_animals(kind, start, end, dst_folder):
#         for i in range(start, end):
#             filename = '{}.{}.jpg'.format(kind, i)
#
#             src_path = os.path.join('DogsandCats/train', filename)
#             dst_path = os.path.join(dst_folder, filename)
#             shutil.copy(src_path, dst_path)
#
#     copy_animals('cat', 0, 1000, 'DogsandCats/small/train/cats')
#     copy_animals('dog', 0, 1000, 'DogsandCats/small/train/dogs')
#     copy_animals('cat', 1000, 1500, 'DogsandCats/small/validation/cats')
#     copy_animals('dog', 1000, 1500, 'DogsandCats/small/validation/dogs')
#     copy_animals('cat', 1500, 2000, 'DogsandCats/small/small/cats')
#     copy_animals('dog', 1500, 2000, 'DogsandCats/small/small/dogs')
def make_small_datasets():
    def copy_animals(kind, start, end, dst_folder):
        for i in range(start, end):
            filename = '{}.{}.jpg'.format(kind, i)

            src_path = os.path.join('DogsandCats/train', filename)
            dst_path = os.path.join(dst_folder, filename)

            shutil.copy(src_path, dst_path)

    copy_animals('cat', 0, 1000, 'DogsandCats/small/train/cats')
    copy_animals('dog', 0, 1000, 'DogsandCats/small/train/dogs')
    copy_animals('cat', 1000, 1500, 'DogsandCats/small/validation/cats')
    copy_animals('dog', 1000, 1500, 'DogsandCats/small/validation/dogs')
    copy_animals('cat', 1500, 2000, 'DogsandCats/small/test/cats')
    copy_animals('dog', 1500, 2000, 'DogsandCats/small/test/dogs')

def generator_basic():
    gen = tf.keras.preprocessing.image.ImageDataGenerator()

    # save_to_dir 옵션 사용하면 폴더에 저장
    flow = gen.flow_from_directory('DogsandCats/small/train',
                                   batch_size=7,
                                   target_size=(224,224),
                                   class_mode='binary',
                                   ) # 문자열로 전달되므로 이름을 정확히 알고 있어야 한다.

    for i, (x, y)  in enumerate(flow):
        print(x.shape, y.shape)
        print(y[:3])
        if i >= 2:
            break

# make_dataset_folders()
# make_small_datasets()
generator_basic()

'''
residual learning

'''

# d = {'name':'kim', 'age':21}
# a = [d, d, d]
# b = [{'name':'kim', 'age':21}] *3 + [{'name':'kim', 'age':21}]
# print(a)
# print(b)

'''
return resnet_utils.Block(scope, bottleneck, [{
    'depth': base_depth * 4,
    'depth_bottleneck': base_depth, 
    'stride': 1  }] * (num_units - 1) + [{
    'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])'''
