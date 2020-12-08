# Day_15_01_sqlite3
import sqlite3


# 1. 데이터 가져오기
# 2. 디비 생성
# 3. 디비에 데이터 추가
# 4. 디비로부터 데이터 가져오기
# 5. ???


# 1. weather.csv 파일을 리스트로 변환해서 반환하는 함수를 만드세요
def get_weather():
    f = open('data/weather.csv', 'r', encoding='utf-8')

    # data = []
    # for row in f:
    #     # print(row)
    #     data.append(row.strip().split(','))
    #     # print(data[-1])

    # 문제
    # 위의 반복문을 컴프리헨션으로 바꾸세요
    data = [row.strip().split(',') for row in f]

    f.close()
    return data


def create_db():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # CREATE TABLE db_list (id INTEGER, name VARCHAR(16));
    query = 'CREATE TABLE kma (prov TEXT, city TEXT, mode TEXT, tmEf TEXT, wf TEXT, tmn TEXT, tmx TEXT, rnSt TEXT)'
    cur.execute(query)

    conn.commit()
    conn.close()


def insert_row(row):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # 문제
    # 데이터를 추가하는 쿼리를 복사해서 함수를 완성하세요
    # INSERT INTO db_list (id, name) VALUES(1, "PC");
    base = 'INSERT INTO kma VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'
    query = base.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
    cur.execute(query)

    conn.commit()
    conn.close()


# 문제
# 전체 데이터를 한번에 추가하는 함수로 수정하세요
def insert_rows(rows):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    base = 'INSERT INTO kma VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'
    for row in rows:
        query = base.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        cur.execute(query)

    conn.commit()
    conn.close()


def show_db():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # 문제
    # 데이터를 읽어오는 명령을 추가하세요
    query = 'SELECT * FROM kma'
    for row in cur.execute(query):
        print(row)

    conn.commit()
    conn.close()


# 문제
# 특정 도시의 데이터를 출력하세요
def search_city(city):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # query = 'SELECT * FROM kma WHERE city=' + city
    query = 'SELECT * FROM kma WHERE city="{}"'.format(city)
    for row in cur.execute(query):
        print(row)

    conn.commit()
    conn.close()


# data = get_weather()
# print(*data, sep='\n')
# print(len(data))

# create_db()

# for row in data:
#     insert_row(row)

# insert_rows(data)

# show_db()

search_city('진천')
search_city('추풍령')

