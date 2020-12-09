# Day_24_01_DogsAndCats.py
import tensorflow as tf
import numpy as np
import os
import shutil

# dogs_and_cats
# +-- train
# +-- small
#     +-- train
#         +-- cat
#         +-- dog
#     +-- validation
#         +-- cat
#         +-- dog
#     +-- test
#         +-- cat
#         +-- dog

# 문제
# 위와 같은 형식의 폴더를 생성하는 함수를 만드세요

# 문제
# 원본 train 폴더의 이미지를
# small/train에 1000개, small/validation에 500개, small/test에 500개 복사하세요
# (파일 이름에 들어있는 규칙을 활용하세요)


def make_dataset_folders():
    def make_if_not(dst_folder):
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

    def make_if_not_2(dst_folder):
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

    # make_if_not('dogs_and_cats/small')
    #
    # make_if_not('dogs_and_cats/small/train')
    # make_if_not('dogs_and_cats/small/validation')
    # make_if_not('dogs_and_cats/small/test')
    #
    # make_if_not('dogs_and_cats/small/train/cat')
    # make_if_not('dogs_and_cats/small/train/dog')
    # make_if_not('dogs_and_cats/small/validation/cat')
    # make_if_not('dogs_and_cats/small/validation/dog')
    # make_if_not('dogs_and_cats/small/test/cat')
    # make_if_not('dogs_and_cats/small/test/dog')

    make_if_not_2('dogs_and_cats/small/train/cat')
    make_if_not_2('dogs_and_cats/small/train/dog')
    make_if_not_2('dogs_and_cats/small/validation/cat')
    make_if_not_2('dogs_and_cats/small/validation/dog')
    make_if_not_2('dogs_and_cats/small/test/cat')
    make_if_not_2('dogs_and_cats/small/test/dog')


# cat.0.jpg
# dog.0.jpg
def make_small_datasets():
    def copy_animals(kind, start, end, dst_folder):
        for i in range(start, end):
            filename = '{}.{}.jpg'.format(kind, i)

            src_path = os.path.join('dogs_and_cats/train', filename)
            dst_path = os.path.join(dst_folder, filename)

            shutil.copy(src_path, dst_path)

    copy_animals('cat', 0, 1000, 'dogs_and_cats/small/train/cat')
    copy_animals('dog', 0, 1000, 'dogs_and_cats/small/train/dog')
    copy_animals('cat', 1000, 1500, 'dogs_and_cats/small/validation/cat')
    copy_animals('dog', 1000, 1500, 'dogs_and_cats/small/validation/dog')
    copy_animals('cat', 1500, 2000, 'dogs_and_cats/small/test/cat')
    copy_animals('dog', 1500, 2000, 'dogs_and_cats/small/test/dog')


def generator_basic():
    gen = tf.keras.preprocessing.image.ImageDataGenerator()

    # save_to_dir 옵션 사용하면 폴더에 저장
    flow = gen.flow_from_directory('dogs_and_cats/small/train',
                                   batch_size=7,
                                   target_size=(224, 224),
                                   class_mode='binary')

    for i, (x, y) in enumerate(flow):
        print(x.shape, y.shape)     # (32, 256, 256, 3) (32, 2)
        print(y[:3])

        if i >= 2:
            break


# make_dataset_folders()
# make_small_datasets()

generator_basic()

# ------------------------------ #

# d = {'name': 'kim', 'age': 21}
# a = [d, d, d]
# # b = [d] * 3
# b = [{'name': 'kim', 'age': 21}] * 3 + [{'name': 'kim', 'age': 21}]
#
# print(a)
# print(b)

# return resnet_utils.Block(scope, bottleneck, [{
#   'depth': base_depth * 4,
#   'depth_bottleneck': base_depth,
#   'stride': 1
# }] * (num_units - 1) + [{
#   'depth': base_depth * 4,
#   'depth_bottleneck': base_depth,
#   'stride': stride
# }])


