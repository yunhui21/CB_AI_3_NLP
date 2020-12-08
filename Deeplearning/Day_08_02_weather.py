# Day_08_02_weather.py
import requests
import re

f = open('data/weather.csv', 'w', encoding='utf-8')

url = 'https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=131'
received = requests.get(url)
# print(received)
# print(received.text)

# 문제
# city만 찾아보세요
# city = re.findall(r'<city>([가-힣]+)</city>', received.text)
# print(city)

# 문제
# location을 찾아보세요

# re.DOTALL : 개행문자 무시. 찾으려고 하는 것이 여러 줄에 걸쳐 있을 때.

# .+ : 탐욕적(greedy)
# .+? : 비탐욕적(non-greedy)
locations = re.findall(r'<location wl_ver="3">(.+?)</location>', received.text, re.DOTALL)
print(len(locations))
# print(locations)

for loc in locations:
    # print(loc)

    # 문제
    # province와 city를 찾아보세요
    # prov = re.findall(r'<province>(.+)</province>', loc)
    # city = re.findall(r'<city>(.+)</city>', loc)
    # print(prov[0], city[0])

    # 문제
    # province와 city를 한번에 찾아보세요 (깔끔하게 출력 포함)
    # prov_city = re.findall(r'<province>(.+)</province>\r\n\t\t\t\t<city>(.+)</city>', loc, re.DOTALL)
    prov_city = re.findall(r'<province>(.+)</province>.+<city>(.+)</city>', loc, re.DOTALL)
    prov, city = prov_city[0]
    # print(prov_city[0])
    # print(prov, city)

    # 문제
    # data를 찾아보세요
    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    # print(len(data))

    # 문제
    # mode, tmEf, wf, tmn, tmx, rnSt를 찾아보세요
    for datum in data:
        # print(datum)
        # mode = re.findall(r'<mode>(.+)</mode>', datum)
        # tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
        # wf = re.findall(r'<wf>(.+)</wf>', datum)
        # tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
        # tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
        # rnSt = re.findall(r'<rnSt>(.+)</rnSt>', datum)
        #
        # print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])

        # 문제
        # mode, tmEf, wf, tmn, tmx, rnSt를 한번에 찾아보세요
        # pattern = r'<mode>(.+)</mode>.+<tmEf>(.+)</tmEf>.+<wf>(.+)</wf>.+<tmn>(.+)</tmn>.+<tmx>(.+)</tmx>.+<rnSt>(.+)</rnSt>'
        # all_info = re.findall(pattern, datum, re.DOTALL)
        # # print(len(all_info))
        #
        # mode, tmEf, wf, tmn, tmx, rnSt = all_info[0]
        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)

        pattern = r'<.+>(.+)</.+>'
        all_info = re.findall(pattern, datum)
        # print(all_info)

        mode, tmEf, wf, tmn, tmx, rnSt = all_info           # unpacking
        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        print(prov, city, *all_info)

        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt, file=f, sep=',')

        # row = '{},{},{},{},{},{},{},{}\n'.format(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        # f.write(row)

        f.write(prov + ',')
        f.write(city + ',')
        f.write(mode + ',')
        f.write(tmEf + ',')
        f.write(wf + ',')
        f.write(tmn + ',')
        f.write(tmx + ',')
        f.write(rnSt + '\n')

f.close()

# 충청북도 옥천
# A02 2020-07-19 00:00 흐림 20 28 40
# A02 2020-07-19 12:00 흐리고 비 20 28 70
# ==>
# 충청북도 옥천 A02 2020-07-19 00:00 흐림 20 28 40
# 충청북도 옥천 A02 2020-07-19 12:00 흐리고 비 20 28 70

# 문제
# 쉽게 읽을 수 있는 형태로 출력을 수정하세요

# 문제
# 기상청 데이터를 파일로 저장하세요 (원본대로 읽을 수 있는 형태로 저장)
# weather.csv (Comma Separated Values)


