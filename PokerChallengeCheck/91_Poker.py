# ------------------------------ #
# 등수: 규정 위반(인터넷 코드 복사. https://www.kaggle.com/kenkpixdev/poker-comb-without-ml-model-accuracy-1-00)
# 캐글: 1.0
# 피씨: 1.0
# ------------------------------ #

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import swifter

path_to_data = '/'

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print(train_data.head())

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# check missing values in graining dataset
print(train_data.isnull().sum())

# Check, how many unique values in our columns
print(train_data.nunique())

f, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
sns.set()
sns.despine(left=True)

sns.countplot(train_data['S1'], color='r', ax=ax[0, 0])
sns.countplot(train_data['C1'], color='b', ax=ax[0, 1])
sns.distplot(train_data['C2'], color='g', kde_kws={'shade': True}, ax=ax[1, 0])
sns.countplot(train_data['CLASS'], color='y', ax=ax[1, 1])
plt.tight_layout()

print(train_data['CLASS'].value_counts())


def pair(row):
    cards = list(row[['C1', 'C2', 'C3', 'C4', 'C5']].values)
    for card in cards:
        if cards.count(card) == 2:
            return True
    return False


def two_pair(row):
    cards = list(row[['C1', 'C2', 'C3', 'C4', 'C5']].values)
    for card in cards:
        if cards.count(card) == 2:
            cards.remove(card)
            for second_pair in cards:
                if cards.count(second_pair) == 2:
                    return True
    return False


def three(row):
    cards = list(row[['C1', 'C2', 'C3', 'C4', 'C5']].values)
    for card in cards:
        if cards.count(card) == 3:
            return True
    return False


def straight(row):
    cards = list(sorted(row[['C1', 'C2', 'C3', 'C4', 'C5']].values))
    need_to_straight = [4, 3, 2, 1, 0]
    straight_to_2 = [1, 10, 11, 12, 13]
    last_card = cards[-1]
    res = []
    if cards == straight_to_2:
        return True
    for card in cards:
        res.append(last_card - card)
    if res == need_to_straight:
        return True
    return False


def flush(row):
    suits = list(row[['S1', 'S2', 'S3', 'S4', 'S5']].values)
    if len(set(suits)) == 1:
        return True
    return False


def full_house(row):
    cards = list(row[['C1', 'C2', 'C3', 'C4', 'C5']].values)
    for card in cards:
        if cards.count(card) == 3:
            check_full_house = list(filter(lambda x: x != card, cards))
            for second_pair in check_full_house:
                if check_full_house.count(second_pair) == 2:
                    return True
    return False


def four_pair(row):
    cards = list(row[['C1', 'C2', 'C3', 'C4', 'C5']].values)
    for card in cards:
        if cards.count(card) == 4:
            return True
    return False


def straight_flush(row):
    cards = list(sorted(row[['C1', 'C2', 'C3', 'C4', 'C5']].values))
    suits = list(row[['S1', 'S2', 'S3', 'S4', 'S5']].values)

    need_to_straight = [4, 3, 2, 1, 0]
    last_card = cards[-1]
    res = []

    if len(set(suits)) == 1:
        for card in cards:
            res.append(last_card - card)
        if res == need_to_straight:
            return True
    return False


def royal(row):
    cards = list(sorted(row[['C1', 'C2', 'C3', 'C4', 'C5']].values))
    suits = list(row[['S1', 'S2', 'S3', 'S4', 'S5']].values)

    need_to_royal = [1, 10, 11, 12, 13]

    if cards == need_to_royal and len(set(suits)) == 1:
        return True
    return False


def poker_combinations(row):
    """
    This function converts information about card in dataset in number of combinations.
    0 - no combinations; 1 - pair; 2 - two pair; 3 - three pair; 4 - straight;
    5 - flush; 6 - full house; 7 - four pair; 8 - straight flush; 9 - royal flush

    Apply this function to train and test dataframe
    """
    if royal(row):
        return 9
    elif straight_flush(row):
        return 8
    elif four_pair(row):
        return 7
    elif full_house(row):
        return 6
    elif flush(row):
        return 5
    elif straight(row):
        return 4
    elif three(row):
        return 3
    elif two_pair(row):
        return 2
    elif pair(row):
        return 1
    else:
        return 0


compare_combinations = train_data.copy()
compare_combinations['combinations'] = compare_combinations.swifter.apply(poker_combinations, axis=1)

for i in range(10):
    print('Class ', i)
    print('True number of samples: ', len(compare_combinations[compare_combinations['CLASS'] == i]), end=', ')
    print('My number of samples: ', len(compare_combinations[compare_combinations['combinations'] == i]))

ssubm = pd.read_csv('data/sample_submission.csv')

test_data = test_data.drop('Id', axis=1)
print(test_data.head())

# Result on test data - 1.00.
# The execution time of this code on test data is 1 hour.
# If you have any suggestions, how to decrease execution time, write this in comments.

n_start = 0
n_features = 25000
n_iterations = int(len(test_data) / n_features)

result = np.array([])

for i in range(n_iterations):
    chunk = test_data[n_start:n_features]
    chunk_res = np.array(chunk.swifter.apply(poker_combinations, axis=1))
    result = np.append(result, chunk_res)
    n_start = n_features
    n_features += 25000

ssubm['CLASS'] = result
ssubm = ssubm.astype('int')
print(ssubm.head())

ssubm.to_csv('outputs/submission_poker.csv', index=False)