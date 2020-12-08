# Day_07_03_re.py
import re

# 3 34 341 3412
db = '''3412    [Bob] 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534'''

# print(db)
numbers = re.findall(r'[0-9]', db)
print(numbers)

print(re.findall(r'[0-9]+', db))

# 문제
# 이름만 찾아보세요 (wrong, just, good)
# print(re.findall(r'[a-Z]+', db))      # a가 Z보다 크니까
print(re.findall(r'[A-z]+', db))        # wrong
print(re.findall(r'[A-Za-z]+', db))     # just
print(re.findall(r'[A-Z][a-z]+', db))   # good

# 문제
# 1. T로 시작하는 이름을 찾으세요
# 2. T로 시작하지 않는 이름을 찾으세요
print(re.findall(r'[T][a-z]+', db))
print(re.findall(r'T[a-z]+', db))

print(re.findall(r'[^T][a-z]+', db))    # ony
print(re.findall(r'[ABCDEFGHIJKLMNOPQRSUVWXYZ][a-z]+', db))
print(re.findall(r'[A-SU-Z][a-z]+', db))

# r : raw
