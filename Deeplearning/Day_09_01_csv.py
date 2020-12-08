# Day_09_01_csv.py
import csv


# 문제
# weather.csv 파일을 읽어서 결과를 반환하세요
def read_csv_1():
    f = open('data/weather.csv', 'r', encoding='utf-8')

    # rows = []
    # for line in f:
    #     # print(line.strip().split(','))
    #     rows.append(line.strip().split(','))

    rows = [line.strip().split(',') for line in f]

    f.close()
    return rows


def read_csv_2():
    f = open('data/weather.csv', 'r', encoding='utf-8')

    rows = []
    for line in csv.reader(f):
        # print(line)
        rows.append(line)

    f.close()
    return rows


def read_us_500():
    f = open('data/us-500.csv', 'r', encoding='utf-8')

    for line in csv.reader(f):
        print(line)

    f.close()


def write_csv(rows):
    f = open('data/kma.csv', 'w', encoding='utf-8', newline='')

    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
    # for row in rows:
    #     writer.writerow(row)

    writer.writerows(rows)

    f.close()


# rows = read_csv_1()
# rows = read_csv_2()

# 문제
# rows를 이전에 출력했던 형태로 출력하세요
# for row in rows:
#     # print(row)
#     for col in row:
#         print(col, end=' ')
#     print()

# for row in rows:
#     for col in row:
#         print(col, end=',')
#     print('\b')

# for row in rows:
#     print(','.join(row))

# read_us_500()

rows = read_csv_2()
write_csv(rows)



