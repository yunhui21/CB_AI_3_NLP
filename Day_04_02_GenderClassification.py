# Day_04_02_GenderClassification.py
import nltk
import random
import string

# nltk.download('names')

# print(nltk.corpus.names.fileids())          # ['female.txt', 'male.txt']
# print(nltk.corpus.names.raw('female.txt'))

# 문제
# 남자와 여자 이름으로 구성된 데이터셋을 반환하는 함수를 만드세요 (셔플 포함)
# 'kim', 'lee', 'han'
# [('kim', 'male'), ('lee', 'female'), ('han', 'female'), ...]


def make_labeled_names():
    m = nltk.corpus.names.words('male.txt')
    f = nltk.corpus.names.words('female.txt')
    # print(len(m), len(f))

    males = [(name.strip(), 'male') for name in m]
    females = [(name.strip(), 'female') for name in f]

    names = males + females
    random.shuffle(names)
    # print(names[:5])

    return names


# 문제
# 남녀 이름의 마지막 글자를 보면, 남자 이름에는 있는데 여자 이름에는 없는 글자가 있습니다.
# 어떤 글자일까요?
def show_omitted(names):
    # males, females = [], []
    # for name, gender in names:
    #     print(name[-1], gender)
    #
    #     # if gender == 'male':
    #     #     males.append(name[-1])
    #     # else:
    #     #     females.append(name[-1])
    #
    #     (males if gender == 'male' else females).append(name[-1])
    #
    # print(males[:5])
    # print(females[:5])

    males, females = set(), set()
    for name, gender in names:
        if gender == 'male':
            males.add(name[-1])
        else:
            females.add(name[-1])

    print(len(males), len(females))
    print(males - females)
    print(females - males)

    # 문제
    # 여자 이름에 들어있는 공백을 없애보세요

    # 문제
    # 남자 이름에 없는 알파벳을 찾으세요
    print(set('abcdefghijklmnopqrstuvwxyz') - males)
    print(set(string.ascii_lowercase) - males)


# 문제
# 검사 데이터에 대해 잘못 분류한 이름들을 알려주세요
def check_mismatch_1(clf, test_set):
    for feature, gender in test_set:
        # print(feature, gender)
        pred = clf.classify(feature)
        print(feature, pred, gender)


def check_mismatch_2(clf, test_names, make_feature):
    males, females = [], []
    for name, gender in test_names:
        # print(name, gender)
        pred = clf.classify(make_feature(name))

        if pred == gender:
            continue

        if pred == 'male':
            males.append(name)
        else:
            females.append(name)

    # 문제
    # 아래 빈 칸에 출력 결과를 표시하는 설명을 붙이세요
    print('여자 이름을 남자로 예측 :', males)
    print('남자 이름을 여자로 예측 :', females)

    # 문제
    # 정확도를 계산하세요
    print('acc :', 1 - (len(males) + len(females)) / len(test_names))

    # 추가 확인 요소
    # 틀린 이름 갯수(원본 갯수), 알파벳에 따른 정확도
    print(len(males))
    print(len(females))


def make_feature_1(name):
    feature = {'last_letter': name[-1]}
    return feature


def make_feature_2(name):
    feature = {'first_letter': name[0], 'last_letter': name[-1]}
    return feature


def make_feature_3(name):
    name = name.lower()
    feature = {'first_letter': name[0], 'last_letter': name[-1]}
    for c in string.ascii_lowercase:
        feature['count_{}'.format(c)] = name.count(c)
        feature['has_{}'.format(c)] = (c in name)

    return feature


def make_feature_4(name):
    feature = {'suffix_1': name[-1], 'suffix_2': name[-2]}
    return feature


def make_feature_5(name):
    feature = {'suffix_1': name[-1], 'suffix_2': name[-2:]} # 26, 26 500개 정도 되는 종류로 구분
    return feature


def make_feature_data(names, make_feature):
    return [(make_feature(name), gender) for name, gender in names]


def gender_basic(make_feature):
    names = make_labeled_names()
    data = make_feature_data(names, make_feature)
    # print(*data[:3], sep='\n')

    train_set, test_set = data[1000:], data[:1000]
    # train_set, test_set = data[:7000], data[7000:]
    # train_set, test_set = data[:-1000], data[-1000:]

    clf = nltk.NaiveBayesClassifier.train(train_set)
    print('acc :', nltk.classify.accuracy(clf, test_set))

    # print(test_set[0])    # ({'last_letter': 'e'}, 'female')

    # 문제
    # Neo와 Trinity에 대해 남자 이름인지 여자 이름인지 알려주세요
    # print(clf.classify({'last_letter': 'o'})) 딕셔너리형태
    # print(clf.classify(make_feature('Neo')))
    # print(clf.classify(make_feature('Trinity')))

    # show_omitted(names)

    # check_mismatch_1(clf, test_set)
    # check_mismatch_2(clf, names[:1000], make_feature)

    # clf.show_most_informative_features(5)
    # clf.show_most_informative_features()      # 기본 10개
    # clf.show_most_informative_features(50)


# 문제
# 5개의 피처 생성 알고리즘의 성능을 객관적으로 비교하세요
def compare_features():
    names = make_labeled_names()

    for make_feature in [make_feature_1, make_feature_2,
                         make_feature_3, make_feature_4, make_feature_5]:
        data = make_feature_data(names, make_feature)
        train_set, test_set = data[1000:], data[:1000]

        clf = nltk.NaiveBayesClassifier.train(train_set)
        print('acc :', nltk.classify.accuracy(clf, test_set))


# gender_basic(make_feature_1)
# gender_basic(make_feature_2)
# gender_basic(make_feature_3)
# gender_basic(make_feature_4)
# gender_basic(make_feature_5)

compare_features()
