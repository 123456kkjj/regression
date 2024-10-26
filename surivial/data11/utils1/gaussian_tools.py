import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture


def split_gaussian(Weight1, Weight2, Miu1, Miu2, delta1, delta2):
    """
    通过高斯密度函数计算两个高斯分量之间的交点
    :param Weight1:  平均值低的高斯分量的权重
    :param Weight2: 平均值搞得高斯分量的权重
    :param Miu1: 平均值低的高斯分量的平均值
    :param Miu2: 平均值搞得高斯分量的平均值
    :param delta1: 平均值低得高斯分量的标准差
    :param delta2: 平均值高得高斯分量的标准差
    :return:
    """

    if not (delta1 and delta2):  #
        return (Miu1 + Miu2) / 2

    A = 1 / (delta1 ** 2) - 1 / (delta2 ** 2)
    B = -2 * (Miu1 / delta1 ** 2 - Miu2 / delta2 ** 2)
    C = Miu1 ** 2 / delta1 ** 2 - Miu2 ** 2 / delta2 ** 2 - 2 * math.log((Weight1 * delta2) / (Weight2 * delta1))
    x1 = (-B + (B ** 2 - 4 * A * C) ** 0.5) / (2 * A)
    x2 = (-B - (B ** 2 - 4 * A * C) ** 0.5) / (2 * A)

    if max(Miu1, Miu2) > x1 > min(Miu1, Miu2):
        return x1
    elif max(Miu1, Miu2) > x2 > min(Miu1, Miu2):
        return x2
    return 0


# def get_gaussian_boundary(data, n_components=5, show=True):
def get_gaussian_boundary(data, n_components=5, show = False):
    """
    通过高斯混合模型得到bic最小的高斯分布，并且画出图
    bic = (-mrna-20wan * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))
    :param data: 数据
    :param n_components: 有多少个混合分量，从1到n_components
    :param show: 是否画图
    :return: 几个高斯分量中的边界
    """
    print('data-37', data.shape)
    X = data.reshape(-1, 1)  # 将数据变成一列

    # X = np.where(np.isnan(X), 0, X)  # 使用0来填充NaN值
    # X = data-37.reshape(-1, 1)  # 将数据变成一列
    # # X = X.fillna(0)
    # X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

    print(X.shape)
    bic = []  # 当前模型在输入X上的贝叶斯信息准则
    best_gmm = GaussianMixture(n_components=1, covariance_type='spherical', random_state=1).fit(X)
    bic.append(best_gmm.bic(X))
    lowest_bic = bic[-1]
    for n_component in range(2, n_components + 1):  # 按照混合组件的数量循环。
        gmm = GaussianMixture(n_components=n_component, covariance_type='spherical', random_state=1).fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm  # 最好的高斯混合模型

    print(best_gmm.n_components)
    means = best_gmm.means_  # 获取每个混合分量的平均值并且做扁平化处理
    means = means.flatten()  # best_gmm.means_得到一个array，再调用np.flatten(),变成1维数组
    # print('means.flatten.shape', means.shape)
    # print('means.flatten', means)

    covars = best_gmm.covariances_  # 每个混合分量的协方差，同上，这个其实是个方差，因为指定的是spherical
    covars = covars.flatten()
    # print('covars.flatten.shape', covars.shape)
    # print('covars.flatten', covars)
    weights = best_gmm.weights_  # 每个混合分量的权重，同上
    weights = weights.flatten()
    # print('weights.flatten.shape', weights.shape)
    # print('weights.flatten', weights)
    x = []
    if len(means) == 1:  # 如果只有一个分量那么返回0
        x.append(0)
        return x

    index_means = np.argsort(means)  # 这个是按照从小到大的means的索引
    for i in range(len(index_means) - 1):
        x.append(split_gaussian(weights[index_means[i]], weights[index_means[i + 1]], means[index_means[i]],
                                means[index_means[i + 1]],
                                covars[index_means[i]] ** 0.5, covars[index_means[i + 1]] ** 0.5))
        # 调用分裂高斯方法，顺序是按照平均值从小到大的顺序

    if show:
        showfigure(weights, means, covars ** 0.5, data, sorted(x))

    # for i in range(len(sorted(x))):
    #     print("x的值为",x[i])

    return sorted(x)

def showfigure(weights, means, covars, data_vector, gaussian_boundary):
    """
    把得到的高斯画图表示出来
    :param weights: 权重
    :param means: 均值
    :param covars: 标准差
    :param data_vector: 数据集
    :param gaussian_boundary: 高斯边界
    :return:
    """
    for i, (w, m, c) in enumerate(zip(weights, means, covars)):  # zip是将对象打包成元组，每一个高斯分量画出一个概率密度曲线
        x = np.arange(min(data_vector), max(data_vector), 0.00001)  # 从最大值到最小值直接按照0.0001为步长生成列表
        y = []
        for j in x:
            y.append(w * _gaussian(j, m, c))  # y代表着概率密度乘以权重

        plt.plot(x, y, '-r')  # x是x轴的数据，y是y轴的数据，x可省略,默认[0,1..,N-1]递增,(x,y1,x,y2) # 此时x不可省略

    for x in gaussian_boundary:
        plt.plot([x, x], [0, 0.2], 'b')  # 画的是高斯边界的线, b代表蓝色

    plt.plot(data_vector, np.zeros(data_vector.shape), '.k', markersize=8)  # k代表黑色, 点的尺寸为8， .代表
    plt.ylabel('Probability density')  # 纵坐标含义，累计重要性
    plt.xlabel('Accumulated importance')  # 横坐标含义，累计重要性
    plt.tick_params(labelsize=18)  # 设置标签尺寸
    plt.show()  # 把图画出来

def _gaussian(x, mean, covar):
    """
    计算高斯分布的概率密度函数
    :param x: 数据点
    :param mean: 均值
    :param covar: 标准差
    :return: 对应的概率密度值
    """
    left = 1 / (np.sqrt(2 * np.pi) * covar)  # 计算分布左侧部分
    right = np.exp(-((x - mean) ** 2) / (2 * covar ** 2))  # 计算指数部分
    return left * right
# def _gaussian(x, mean, covar):
#     """
#     算的其实是概率密度函数,具体公式查看百度百科的高斯分布的概率密度公式。
#     :param x: 梯度值
#     :param mean: 平均值
#     :param covar: 标准差
#     :return:
#     """
#     left = 1 / (math.sqrt(2 * math.pi) * covar)
#     right = math.exp(-math.pow(x - mean, 2) / (2 * math.pow(covar, 2)))  # exp() 方法返回x的指数,e^x,
#     # pow() 方法返回 x^y（x 的 y 次方）的值。
#
#     return left * right
