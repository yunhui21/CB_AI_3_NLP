# ------------------------------ #
# 등수: 규정 위반(머신러닝/딥러닝 코드 없이 프로그래밍으로만 해결)
# 캐글: 1.0
# 피씨: 1.0
# ------------------------------ #

import pandas as pd
import numpy as np

# 데이터 준비
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
print(df_train.shape, df_test.shape)

print(df_train.head())
print(df_test.head())

df_train = df_train.drop(['Id'], axis=1)
df_test= df_test.drop(['Id'], axis=1)
print(df_train.shape, df_test.shape)

# 빈 값 개수 찾기 : isnull().sum()
print(df_train.isnull().sum())

# 각 열에서 고유한 값 개수 : nunique()
print(df_train.nunique())

# 'CLASS' 열에 클래스별 데이터 개수
#   8번 클래스 비어있음
#   unbalanced data set
print(df_train['CLASS'].value_counts())

# 학습 및 테스트 데이터 세트 설정
#   x_train, y_train
#   x_test
x_train = df_train[['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5']]
print(x_train.head())

y_train = df_train[['CLASS']]
print(y_train.head())

x_test = df_test[['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5']]
print(x_test.head())

# 로직 함수 정의
# 포커핸드에 예외가 없으므로 머신러닝으로 하는 것 보다는 논리함수 사용하여 예측

# 11) CLASS �Poker Hand� Ordinal (0-9)
    # 0: Nothing in hand; not a recognized poker hand
    # 1: One pair; one pair of equal ranks within five cards
    # 2: Two pairs; two pairs of equal ranks within five cards
    # 3: Three of a kind; three equal ranks within five cards
    # 4: Straight; five cards, sequentially ranked with no gaps
    # 5: Flush; five cards with the same suit
    # 6: Full house; pair + different rank three of a kind
    # 7: Four of a kind; four equal ranks within five cards
    # 8: Straight flush; straight + flush
    # 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush


def assign_label(row):
    suits = list(row[['S1', 'S2', 'S3', 'S4', 'S5']].values)
    ranks = list(row[['C1', 'C2', 'C3', 'C4', 'C5']].values)

    count_eq = 0
    count_seq = 0

    n_suit = len(set(suits))
    n_rank = len(set(ranks))

    srt_rank = sorted(ranks)
    royal_rank = [1, 13, 12, 11, 10]

    for i in range(1, 5):
        if (srt_rank[i] == srt_rank[i - 1] + 1):
            count_seq += 1

    # 9: Royal flush; {Ace, King, Queen, Jack, Ten} + Flush; five cards with the same suit
    if n_suit == 1:
        if (srt_rank == sorted(royal_rank)):
            return 9

        # 8: Straight flush;
        #    Straight; five cards, sequentially ranked with no gaps + Flush; five cards with the same suit
        elif (count_seq == 4):
            return 8

        # 5: Flush; five cards with the same suit
        else:
            return 5

    # 4: Straight; five cards, sequentially ranked with no gaps
    if (count_seq == 4) or (srt_rank == sorted(royal_rank)):
        return 4

    for i in range(1, 5):
        if (srt_rank[i] == srt_rank[i - 1]):
            count_eq += 1

            # # 0: Nothing in hand; not a recognized poker hand
    if (count_eq == 0) and (count_seq < 4):
        return 0

    # 1: One pair; one pair of equal ranks within five cards
    elif count_eq == 1:
        return 1

    # 2: Two pairs; two pairs of equal ranks within five cards
    # 3: Three of a kind; three equal ranks within five cards
    elif count_eq == 2:

        count = [1, 0, 0]
        j = 0
        for i in range(1, 5):
            if (srt_rank[i] == srt_rank[i - 1]):
                count[j] += 1
            else:
                j += 1
                count[j] += 1

        count = sorted(count)

        if count[2] == 2:
            return 2
        if count[2] == 3:
            return 3

    # 7: Four of a kind; four equal ranks within five cards
    # 6: Full house;
    #    One pair; one pair of equal ranks within five cards + different rank three of a kind
    elif n_rank == 2:
        if (srt_rank[0] != srt_rank[1]):
            return 7
        elif (srt_rank[1] == srt_rank[2] == srt_rank[3]):
            return 7
        else:
            return 6


# 예측이 틀린 경우 확인
train_pred = []
for i in range(len(x_train)):
    # for i in range(100, 200):
    x_row = x_train.iloc[i]
    y_row = y_train.iloc[i][0]
    hand_num = assign_label(x_row)
    # print("{}. True Value: {} and Predicted Value: {}".format(i, y_row, hand_num))

    if (y_row != hand_num):
        print(x_row[['S1', 'S2', 'S3', 'S4', 'S5']].values)
        print(x_row[['C1', 'C2', 'C3', 'C4', 'C5']].values)
        print("{}. True Value: {} and Predicted Value: {}".format(i, y_row, hand_num))

# test 데이터를 이용한 예측
test_pred = []
for i in range(len(x_test)):
    # for i in range(100):
    row = x_test.iloc[i]
    hand_num = assign_label(row)
    test_pred.append(hand_num)

# submission 파일 작성
#   sample_submission 파일 읽어오기
sub = pd.read_csv('data/sample_submission.csv')
print(sub.head())

# submission 파일 예측 값으로 수정 및 저장
submission = sub[['Id', 'CLASS']]
submission['CLASS'] = test_pred
submission.to_csv('outputs/submission_YongBokLee.csv', index=False)
# # 파일 수정 여부 확인
# sub = pd.read_csv('outputs/submission_YongBokLee.csv')
# sub