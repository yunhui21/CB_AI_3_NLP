# Day_15_02_songs.py
import requests
import re


def show_songs(code, page):
    payload = {
        'S_PAGENUMBER': page,       # 1
        'S_MB_CD': code,            # 'W0726200'
        'S_HNAB_GBN': 'I',
        'hanmb_nm': 'G-DRAGON',
        'sort_field': 'SORT_PBCTN_DAY'
    }

    # 문제
    # 한국음악저작권협회에서 지드래곤의 음악 데이터를 가져오세요
    url = 'https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp'
    received = requests.post(url, data=payload)
    # print(received.text)

    # https://www.google.com/search   ?   q=python  &  oq=python

    # get/post
    # post
    # 1. 암호화
    # 2. 대용량
    # 3. 폼 데이터 전

    # 문제
    # 지드래곤의 노래를 가져와 보세요
    tbody = re.findall(r'<tbody>(.+?)</tbody>', received.text, re.DOTALL)
    # print(len(tbody))
    # print(tbody[1])

    tbody_text = tbody[1]

    # imgs = re.findall(r'<img src="/images/common/control.gif"  alt="" />', tbody_text)
    # print(len(imgs))

    # imgs = re.findall(r'<img .+? />', tbody_text)
    # print(len(imgs))

    tbody_text = re.sub(r' <img .+? />', '', tbody_text)
    tbody_text = re.sub(r'<br/>', ',', tbody_text)

    trs = re.findall(r'<tr>(.+?)</tr>', tbody_text, re.DOTALL)
    # print(len(trs))
    # print(trs[0])

    if not trs:
        return False

    # 문제
    # td 데이터를 검색하세요
    for tr in trs:
        # print(tr)
        tds = re.findall(r'<td>(.+)</td>', tr)
        # tds = [td.strip() for td in tds]
        tds[0] = tds[0].strip()
        print(tds)

    return True

    # 문제
    # <br/> 태그를 쉼표로 바꾸세요

    # 문제
    # 제목 앞에 있는 빈 칸을 삭제하세요


# show_songs('W0726200', 10000)

page = 1
while show_songs('W0726200', page):
    print('---------------', page)
    page += 1


# 봇 bot
