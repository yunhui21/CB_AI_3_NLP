# Day_09_02_open_hangul.py
import requests
import re


# 문제
# 오픈한글 사이트에서 제공하는 API를 사용해서 한글의 영문 자판을 반환하는 함수를 만드세요
def kor2eng(kor):
    url = 'https://openhangul.com/nlp_ko2en?q=' + kor
    received = requests.get(url)
    # print(received.text)

    pattern = '<img src="images/cursor.gif"><br>(.+)</pre>'
    eng = re.findall(pattern, received.text, re.DOTALL)
    return eng[0].strip()


print(kor2eng('꼬깔콘'))
print(kor2eng('한글'))
print(kor2eng('손'))


