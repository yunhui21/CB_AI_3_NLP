# Day_33_01_tf_datasets.py
import tensorflow_datasets as tfds


def tfds_basic_1():
    names = tfds.list_builders()
    # print(names)        # ['abstract_reasoning', 'accentdb', 'aeslc', 'aflw2k3d'...]
    # print(len(names))   # 231

    imdb_1 = tfds.load('imdb_reviews')
    # print(type(imdb_1))     # <class 'dict'>
    # print(imdb_1.keys())    # ['test', 'train', 'unsupervised']
    # print(imdb_1)
    # #{'test': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>,
    # # 'train': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>,
    # # 'unsupervised': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>}
    #
    # print(type(imdb_1['train'])) # <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
    # print(imdb_1['train'])  # <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>
    print('-'*50)

    # for take in imdb_1['train'].take(2):
    #     print(type(take), take.keys())      # <class 'dict'> dict_keys(['label', 'text'])
    #     print(take['label'].numpy())        # 0
    #     print(take['text'].numpy())         # b'I have been known to fall
    #     print()

    # for take in imdb_1['train'].take(2).as_numpy_iterator():
    #     print(take)
    #     print(take['label'], take['text'])
    #     print()

def tfds_basic_2():
    imdb_2, info = tfds.load('imdb_reviews', with_info=True)
    print(info)

    # splits 속성값
    print(info.splits)
    # {'test': <tfds.core.SplitInfo num_examples=25000>,
    # 'train': <tfds.core.SplitInfo num_examples=25000>,
    # 'unsupervised': <tfds.core.SplitInfo num_examples=50000>}

    print(info.splits.keys())   # dict_keys(['test', 'train', 'unsupervised'])
    print(info.splits['train'])                     # <tfds.core.SplitInfo num_examples=25000>
    print(info.splits['train'].num_examples)        # 25000

    # imdb_2를 통해서 바로 보는작업
    print(imdb_2['train'].cardinality())            # tf.Tensor(25000, shape=(), dtype=int64)
    print(imdb_2['train'].cardinality().numpy())    # 25000
    print('-'*30)


    print('-'* 50)

    train_set, test_set = tfds.load('imdb_reviews', split=['train', 'test'])
    print(type(train_set))

    # 문제
    # 리뷰 데이터를 train, validation, test로 나눠주세요.
    train_set,valid_set, test_set = tfds.load(
        'imdb_reviews',
        as_supervised=True,
        # split=['train[:60%]', 'train[:60%]', 'test']

    )

    # 문제
    # train, validation, test 데이터의 개수를 출력하세요.
    print(train_set.cardinality().numpy())
    print(valid_set.cardinality().numpy())
    print(test_set.cardinality().numpy())

    # print(type(train_set), len(train_set))
    # print(type(valid_set), len(valid_set))
    # print(type(test_set), len(test_set))

def tfds_basic_3():
    imdb_3 = tfds.load('imdb_reviews', split=['train', 'test'])
    print(type(imdb_3), len(imdb_3))  # <class 'list'> 2
    print(imdb_3[
              0])  # <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}> - dict 형태로 관리

    for take in imdb_3[0].take(2):
        print(type(take))  # <class 'dict'>
        print(take['label'], take['text'])
    print()

    imdb_3 = tfds.load('imdb_reviews', split=['train', 'test'])
    print(type(imdb_3), len(imdb_3))  # <class 'list'> 2
    print(imdb_3[0])  # <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)> - tuple의 형태로 관리

    for take in imdb_3[0].take(2):
        print(type(take))
        print(take[0], take[1])

    for xx, yy in imdb_3[0].take(2):
        print(xx.numpy(), yy.numpy())


# tfds_basic_1()
# tfds_basic_2()
# tfds_basic_3()
