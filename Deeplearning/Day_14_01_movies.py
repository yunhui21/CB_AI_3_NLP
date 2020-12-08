# Day_14_01_movies.py
import pandas as pd
import numpy as np


# pk : primary key
# fk : foreign key

# 문제
# 여성들이 선호하는 영화 5개를 알려주세요

# 1. 영화 제목을 인덱스로 하는 데이터프레임 생성
# 2. 500번 이상 평가된 영화 제목 데이터프레임으로 변환
# 3. 여성 평점이 제일 높은 5개의 영화 선정


def get_data():
    users = pd.read_csv('ml-1m/users.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv('ml-1m/movies.dat',
                         header=None,
                         sep='::',
                         engine='python',
                         names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('ml-1m/ratings.dat',
                          header=None,
                          sep='::',
                          engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    # print(users)
    # print(movies)
    # print(ratings)

    data = pd.merge(pd.merge(ratings, users), movies)
    # print(data)

    return data


def pivot_basic():
    df = get_data()

    pv1 = df.pivot_table(index='Age', values='Rating')
    print(pv1, end='\n\n')

    pv2 = df.pivot_table(index='Age', columns='Gender', values='Rating')
    print(pv2, end='\n\n')

    # 문제
    # 18세 구간의 여성 데이터를 가져오세요 (2가지)
    # print(pv2.F, end='\n\n')
    print(pv2.F[18], end='\n\n')

    # print(pv2.loc[18], end='\n\n')
    print(pv2.loc[18]['F'], end='\n\n')

    pv3 = df.pivot_table(index=['Age', 'Gender'], values='Rating')
    print(pv3, end='\n\n')
    print(pv3.unstack(), end='\n\n')
    print(pv3.unstack().stack(), end='\n\n')

    # 문제
    # 18세 구간의 여성 데이터를 가져오세요 (2가지)
    print(pv3.loc[18], end='\n\n')          # <class 'pandas.core.frame.DataFrame'>
    # print(pv3.loc[18]['F'], end='\n\n')
    print(type(pv3.loc[18]), end='\n\n')
    print(pv3.loc[18].loc['F'], end='\n\n')
    print(type(pv3.loc[18].loc['F']), end='\n\n')

    print(pv3.loc[18].loc['F'][0], end='\n\n')
    print(pv3.loc[18, 'F'], end='\n\n')

    pv4 = df.pivot_table(index='Age', columns=['Occupation', 'Gender'], values='Rating')
    print(pv4, end='\n\n')

    pv5 = df.pivot_table(index='Age', columns=['Occupation', 'Gender'],
                         values='Rating', fill_value=0)
    print(pv5, end='\n\n')

    pv6 = df.pivot_table(index='Age', columns='Gender', values='Rating',
                         aggfunc=[np.mean, np.sum])
    print(pv6, end='\n\n')


def get_index_500():
    # 문제
    # 1. 영화 제목을 인덱스로 하는 데이터프레임 생성 (by_title)
    df = get_data()

    pv_title = df.pivot_table(index='Title', columns='Gender', values='Rating')
    print(pv_title, end='\n\n')

    # 2. 500번 이상 평가된 영화 제목 데이터프레임으로 변환
    # by_title = df.groupby(by='Title')
    #
    # for item in by_title:
    #     print(item)

    by_title = df.groupby(by='Title').size()
    print(by_title, end='\n\n')

    # by_bools = (by_title.values >= 500)
    # print(by_bools)

    by_bools = (by_title >= 500)        # broadcast
    print(by_bools, end='\n\n')

    index_500 = pv_title[by_bools]
    print(index_500, end='\n\n')

    return index_500


# pivot_basic()

index_500 = get_index_500()

# 3. 여성 평점이 제일 높은 5개의 영화 선정
# top_female = index_500.sort_values('F')
top_female = index_500.sort_values('F', ascending=False)
print(top_female.head(), end='\n\n')
# print(top_female.head(5), end='\n\n')

print(top_female[:5], end='\n\n')

# 문제
# 남녀 호불호가 갈리지 않는 영화 5개를 알려주세요
index_500['Diff'] = (index_500.F - index_500.M).abs()       # vector
print(index_500, end='\n\n')

diff_500 = index_500.sort_values('Diff')
print(diff_500.head(), end='\n\n')

