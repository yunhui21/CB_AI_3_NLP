# Day_07_02_json.py
import json
import requests
import re

d1 = {"ip": "8.8.8.8"}
print(d1['ip'])
print(d1)
print(type(d1))

d2 = json.dumps(d1)
# print(d2['ip'])       # error
print(d2)
print(type(d2))

d3 = json.loads(d2)
print(d3)
print(type(d3))
print('-' * 50)

# 문제
# 아래 데이터에서 값에 해당하는 부분만 출력하세요
dt = '{"time": "03:53:25 AM", "epoch": 1362196405309, "date": "03-02-2013"}'
dt2 = json.loads(dt)
print(dt2)

print(dt2["time"], dt2["epoch"], dt2['date'])
print('-' * 50)

url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'

received = requests.get(url)
print(received)
print(received.text)

origin = received.content
print(origin)
print(type(origin))

text = bytes.decode(origin, encoding='utf-8')  # euc-kr
print(text)
# [{"code":"11","value":"서울특별시"},
# {"code":"26","value":"부산광역시"},
# {"code":"27","value":"대구광역시"},
# {"code":"28","value":"인천광역시"},
# {"code":"29","value":"광주광역시"},
# {"code":"30","value":"대전광역시"},
# {"code":"31","value":"울산광역시"},
# {"code":"41","value":"경기도"},
# {"code":"42","value":"강원도"},
# {"code":"43","value":"충청북도"},
# {"code":"44","value":"충청남도"},
# {"code":"45","value":"전라북도"},
# {"code":"46","value":"전라남도"},
# {"code":"47","value":"경상북도"},
# {"code":"48","value":"경상남도"},
# {"code":"50","value":"제주특별자치도"}]
# print(text[:10])
# print(text[10:12])

# 문제
# code와 value에 들어있는 값만 출력하세요 (반복문)
items = json.loads(text)
print(items)
print(type(items))
print(items[0])
print(items[0]['code'])
print(items[1]['code'])

for item in items:
    # print(item)
    print(item['code'], item['value'])

print([(item['code'], item['value']) for item in items])
print([print('hello'), print('hello'), print('hello'), ])


# 문제
# 정규표현식을 사용해서 code와 value만 출력하세요
codes = re.findall(r'[0-9]+', text)
values = re.findall(r'[가-힣]+', text)
print(codes)
print(values)

# for i in range(len(codes)):
#     print(codes[i], values[i])

binds = zip(codes, values)
print(binds)

# print(next(binds))
# print(next(binds))
# print(next(binds))

# print(list(binds))

# for item in binds:
#     print(item)

for c, v in binds:
    print(c, v)
print('-' * 50)

# [{"code":"11","area":"0.5","value":"서울특별시"},{"code":"26","area":"0.7","value":"부산광역시"},
print(re.findall(r'"code":"11"', text))
print(re.findall(r'"code":"[0-9]+"', text))
print(re.findall(r'"code":"([0-9]+)"', text))

# 문제
# 시도 이름만 출력하세요
print(re.findall(r'"value":"[가-힣]+"', text))
print(re.findall(r'"value":"([가-힣]+)"', text))

# 문제
# code와 value를 한번에 찾으세요
print(re.findall(r'{"code":"([0-9]+)","value":"([가-힣]+)"}', text))


