# Day_12_02_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, rc


# 문제
# 남자와 여자 막대 그래프를 그리세요
def bar_1():
    men = [25, 19, 21, 23, 29]
    women = [24, 28, 16, 31, 25]

    indices = np.arange(5)

    # plt.bar(indices, men)
    plt.bar(indices, men, width=0.45)
    plt.bar(indices + 0.5, women, width=0.45)

    plt.xticks(indices + 0.5 / 2, ['A', 'B', 'C', 'D', 'E'])
    plt.show()


# 문제
# GDP 파일로부터 상위 10개 나라의 데이터를 막대 그래프로 그려보세요
def bar_2():
    f = open('data/2016_GDP.txt', 'r', encoding='utf-8')

    # skip header
    f.readline()

    names, dollars = [], []
    for row in f:
        # items = row.strip().split(':')
        # print(items)
        #
        # names.append(items[1])
        # dollars.append(items[2])

        _, name, dollar = row.strip().split(':')    # _ : place holder

        dollar = dollar.replace(',', '')

        names.append(name)
        dollars.append(int(dollar))

    f.close()

    # ----------------------- #

    names_10 = names[:10]
    dollars_10 = dollars[:10]

    print(names_10)
    print(dollars_10)

    indices = range(10)

    # ttf = 'C:/Windows/Fonts/malgun.ttf'
    ttf = '/Library/Fonts/Arial Unicode.ttf'
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)

    # plt.bar(indices, dollars_10)
    # plt.bar(indices, dollars_10, color='r')
    # plt.bar(indices, dollars_10, color='rgb')
    # plt.bar(indices, dollars_10, color=['red', 'green', 'blue'])
    # plt.bar(indices, dollars_10, color=colors.BASE_COLORS)
    plt.bar(indices, dollars_10, color=colors.TABLEAU_COLORS)

    # plt.xticks(indices, names_10)
    # plt.xticks(indices, names_10, rotation='vertical')
    # plt.xticks(indices, names_10, rotation=270)
    plt.xticks(indices, names_10, rotation=45)

    plt.title('2016 GDP')
    plt.xlabel('나라 이름')

    plt.subplots_adjust(bottom=0.2)
    plt.show()


# bar_1()
bar_2()




