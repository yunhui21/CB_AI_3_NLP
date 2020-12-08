# ------------------------------ #
# 등수: 2
# 캐글: 0.99998
# 피씨: 0.99998
# ------------------------------ #

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier

# import os
# for dirname, _, filenames in os.walk('/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')

train = train_raw.loc[:,train_raw.columns != 'Id']
X_train = train.loc[:,train.columns != 'CLASS']
Y_train = train["CLASS"]

X_test = test_raw.loc[:,test_raw.columns != 'Id']
X_id = test_raw["Id"]
print(train.head())


def sortnumber(data):
    df = data.copy()
    temp = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    temp.values.sort()
    df[['N1', 'N2', 'N3', 'N4', 'N5']] = temp
    df = df[['N1', 'N2', 'N3', 'N4', 'N5', 'S1', 'S2', 'S3', 'S4', 'S5']]
    return df


def diff_suit_count(df):
    temp = df[['S1', 'S2', 'S3', 'S4', 'S5']]
    df['DS'] = temp.apply(lambda i: len(np.unique(i)) , axis=1)
    # unique 함수는 서로 다른 숫자를 리스트로 반환합니다. 람다식을 행에 적용하였으므로 한 카드 패에서 서로 다른 무늬가 몇 개인지 알아냅니다.


def sub(df):
    df['sub54'] = df['N5'] - df['N4']
    df['sub43'] = df['N4'] - df['N3']
    df['sub32'] = df['N3'] - df['N2']
    df['sub21'] = df['N2'] - df['N1']


X_train_pre = sortnumber(X_train)
diff_suit_count(X_train_pre)
sub(X_train_pre)

X_test_pre = sortnumber(X_test)
diff_suit_count(X_test_pre)
sub(X_test_pre)

print(X_train_pre.head())
print(X_test_pre.head())

print(X_train_pre.shape)

model = DecisionTreeClassifier(random_state=3, criterion='gini')
model.fit(X_train_pre, Y_train)

# 어떤 일정한 규칙을 정의하기 쉬운 데이터이기 때문에 결정트리 분류기가 굉장히 효율적으로 작동할 수 있습니다.

y_pred = model.predict(X_test_pre)

pred = y_pred.flatten()
pred_list=list(pred)
result = X_id.to_frame()
result = result.assign(CLASS=pred_list)

result.to_csv("outputs/submission_smashh712.csv", index=False)
print(result)



