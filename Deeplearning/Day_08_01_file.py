# Day_08_01_file.py
import re


def read_1():
    f = open('data/poem.txt', 'r', encoding='utf-8')

    lines = f.readlines()
    print(lines)

    f.close()


def read_2():
    f = open('data/poem.txt', 'r', encoding='utf-8')

    while True:
        line = f.readline()
        # print(len(line))

        # 거짓 : 0, 0.0, False, None, [], ''
        # if len(line) == 0:
        if not line:
            break

        # print(line, end='')
        print(line.strip())     # 양쪽 공백들 제거   '   he ll oo  '

    f.close()


def read_3():
    f = open('data/poem.txt', 'r', encoding='utf-8')

    # lines = []
    # for line in f:
    #     # print(line.strip())
    #     lines.append(line.strip())
    lines = [line.strip() for line in f]

    f.close()
    return lines


def read_4():
    with open('data/poem.txt', 'r', encoding='utf-8') as f:
        for line in f:
            print(line.strip())


# 문제
# lines에 들어있는 단어의 갯수를 출력하세요
def show_word_count(lines):
    count = 0
    for line in lines:
        print(line)
        words = re.findall(r'[가-힣]+', line)
        print(words)

        count += len(words)

    print('words :', count)


def write():
    f = open('data/sample.txt', 'w', encoding='utf-8')

    f.write('hello\n')
    f.write('python')

    f.close()


# read_1()
# read_2()
# read_3()
# read_4()

# lines = read_3()
# print(lines)

# show_word_count(lines)

write()
