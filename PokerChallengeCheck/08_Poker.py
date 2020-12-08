# ------------------------------ #
# 등수: 8
# 캐글: 0.99740
# 피씨: 0.99
# ------------------------------ #

# !pip3 install iterative-stratification
import tensorflow.keras
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

train['CLASS']=train['CLASS'].replace(9,8)
train['CLASS'].unique()

alldata=pd.concat([train,test],sort=False)
alldata=alldata.drop(['Id','CLASS'],axis=1)
alldata.describe()

train2=alldata[:len(train)].astype(float)
test2=alldata[len(train):]

y = train.iloc[:,-1]
y = tensorflow.keras.utils.to_categorical(y, 9)
test = test.iloc[:,1:]

# kfold로 인한 출력량과 시간 소비 때문에 주석 처리로 대체 합니다.
reLR = ReduceLROnPlateau(patience=300,verbose=1,factor=0.65)
mskf = MultilabelStratifiedKFold(n_splits=20, shuffle=True, random_state=0)
result=0
loop = 0
for train_index, valid_index in mskf.split(train2, y):
    loop +=1
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~{}번째~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(loop))
    x_train=train2.iloc[train_index]
    x_valid=train2.iloc[valid_index]
    y_train=y[train_index]
    y_valid=y[valid_index]
    es=EarlyStopping(patience=550,verbose=1)
    mc=ModelCheckpoint("outputs/v2_980_k20_{}.h5".format(loop),save_best_only=True,verbose=1)

    model=Sequential()

    model.add(Dense(3140,activation='relu',input_dim=train2.shape[1]))
    model.add(Dense(3140,activation='relu'))

    model.add(Dense(2344,activation='relu'))
    model.add(Dense(2344,activation='relu'))

    model.add(Dense(1532,activation='relu'))
    model.add(Dense(1532,activation='relu'))

    model.add(Dense(730,activation='relu'))
    model.add(Dense(730,activation='relu'))

    model.add(Dense(330,activation='relu'))
    model.add(Dense(330,activation='relu'))

    model.add(Dense(130,activation='relu'))
    model.add(Dense(130,activation='relu'))

    model.add(Dense(30,activation='relu'))
    model.add(Dense(30,activation='relu'))

    model.add(Dense(9,activation = 'softmax'))

    model.compile(metrics=['acc'], loss = 'categorical_crossentropy', optimizer='adam')
    model.fit(x_train,y_train,epochs=5000,validation_data=(x_valid,y_valid),callbacks=[es,mc,reLR],batch_size=256)

    model.load_weights("./v2_980_k20_{}.h5".format(loop))
    result+=model.predict(test)
result /= 20
pd.DataFrame(result).to_csv('outputs/submission_k_roll.csv',index=False)

sub['CLASS'] = 0
sub['CLASS'] = result.argmax(axis=1)
sub['CLASS'] = sub['CLASS'].replace(8,9)
sub['CLASS'].value_counts()

sub.to_csv('outputs/submission_k_roll',index=False)