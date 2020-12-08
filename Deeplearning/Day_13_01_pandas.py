# Day_13_01_pandas.py
import pandas as pd

s1 = pd.Series([2, 1, 4, 9])
print(s1)
print(type(s1))

print(s1.index)         # RangeIndex(start=0, stop=4, step=1)
print(s1.values)        # [2 1 4 9]
print(type(s1.values))  # <class 'numpy.ndarray'>

print(s1[0], s1[3])
# print(s1[-1])
print('-' * 30)

s2 = pd.Series([2, 1, 4, 9], index=['a', 'b', 'c', 'd'])
print(s2)

# 문제
# 숫자 4를 출력하는 두 가지 코드를 만드세요
print(s2[2])
print(s2['c'])
print(s2[-2])

# 문제
# 슬라이싱을 사용해서 가운데 2개를 출력하세요 (2가지)
print(s2[1:-1])
print(s2['b':'c'])

print(s2[1:-1].values)
print(s2.values[1:-1])
print('-' * 30)

df = pd.DataFrame({
    'year': [2018, 2019, 2020, 2018, 2019, 2020],
    'city': ['ochang', 'ochang', 'ochang', 'sejong', 'sejong', 'sejong'],
    'rain': [130, 135, 145, 140, 150, 145],
})

print(df)
print(type(df))

print(df.head(), end='\n\n')
print(df.tail(), end='\n\n')

print(df.head(2), end='\n\n')
print(df.tail(2), end='\n\n')

df.info()

print(df.index)
print(df.columns)
print(df.values)
print(df.values.dtype, end='\n\n')

print(df['year'], end='\n\n')
print(df.year, end='\n\n')
print(type(df.year), end='\n\n')

df.index = ['a', 'b', 'c', 'd', 'e', 'f']
print(df, end='\n\n')

print(df.loc['a'], end='\n\n')
print(df.iloc[0], end='\n\n')

print(df.loc['f'], end='\n\n')
print(df.iloc[-1], end='\n\n')

# 문제
# 데이터프레임에서 마지막 3행을 출력하세요 (2가지)
print(df.iloc[3:], end='\n\n')
print(df.loc['d':], end='\n\n')

print(df[3:], end='\n\n')
print('-' * 30)

#               index  columns values
print(df.pivot('year', 'city', 'rain'))
