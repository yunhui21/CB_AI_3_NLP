# Day_13_03_colormap.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd


def colormap_1():
    x = np.random.rand(100)
    y = np.random.rand(100)
    t = np.arange(100)

    print(x)

    # plt.plot(x, y)
    # plt.plot(x, y, 'ro')
    plt.scatter(x, y, c=t)
    plt.show()


# 문제
# scatter 함수를 이용해서 대각선을 2개 그려보세요
def colormap_2():
    x = np.arange(100)

    print(cm.viridis(0))
    print(cm.viridis(255))

    plt.scatter(x, x, c=x)
    # plt.scatter(x, x[::-1], c=x)
    # plt.scatter(x[::-1], x, c=x, cmap='jet')
    plt.scatter(x[::-1], x, c=x, cmap='viridis')
    plt.show()


def colormap_3():
    print(plt.colormaps())
    print(len(plt.colormaps()))         # 164
    # ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
    # 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
    # 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
    # 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
    # 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    # 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
    # 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    # 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    # 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
    # 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r',
    # 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r',
    # 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
    # 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
    # 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
    # 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r',
    # 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
    # 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    # 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
    # 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r',
    # 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r',
    # 'winter', 'winter_r']

    x = np.arange(100)

    plt.subplot(1, 2, 1)
    plt.scatter(x, x, c=x, cmap='gist_rainbow')

    plt.subplot(1, 2, 2)
    plt.scatter(x, x, c=x, cmap='gist_rainbow_r')

    plt.colorbar()
    plt.show()


def colormap_4():
    jet = cm.get_cmap('jet')

    print(jet(-5))
    print(jet(0))
    print(jet(128))
    print(jet(255))
    print(jet(256))
    print()

    print(jet(0.1))
    print(jet(0.5))
    print(jet(128/255))
    print(jet(0.9))
    print()

    print(jet([0, 255]))
    print(jet(range(0, 256, 64)))
    print(jet(np.linspace(0.2, 0.5, 7)))

    # print(np.arange(0, 1, 0.1))
    # print(np.linspace(0, 1, 11))


def colormap_5():
    n = np.random.rand(10, 10)
    print(n)

    # plt.imshow(n)
    plt.imshow(n, cmap='winter')
    plt.show()


def colormap_6():
    flights = sns.load_dataset('flights')
    print(flights)

    df = flights.pivot('month', 'year', 'passengers')
    print(df)

    # plt.figure(1, figsize=[16, 8])

    # plt.pcolor(df)
    # # plt.xticks(np.arange(12), df.columns)
    # plt.xticks(0.5 + np.arange(0, 12, 2), df.columns[::2])
    # plt.yticks(0.5 + np.arange(12), df.index)
    # plt.title('flights heatmap')
    # plt.colorbar()

    # sns.heatmap(df)
    # sns.heatmap(df, annot=True, fmt='d')
    sns.heatmap(df, annot=True, fmt='d', cmap='viridis')

    plt.show()


# colormap_1()
# colormap_2()
# colormap_3()
# colormap_4()
# colormap_5()
colormap_6()



