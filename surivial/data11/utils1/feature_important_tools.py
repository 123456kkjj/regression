import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# def load_feature_impts_from_dir(file_dir=r'save_vimps', show=False):
#     '''
#     导入基因重要性特征
#     '''
#
#     file_list = os.listdir(file_dir)
#     for i, file in enumerate(file_list):
#         file_path = os.path.join(file_dir, file)
#         if i == 0:
#             data = np.load(file_path)
#         else:
#             data += np.load(file_path)
#
#     plt.plot(np.arange(data.shape[0]), data[:, 1], '.k')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Checked feature Count')
#     plt.title('Feature Count')
#     plt.show()
#
#     # plt.hist(data[:,1])
#     # plt.show()
#
#     vimps = data[:, 0] / data[:, 1]  # 得分，越小越重要
#
#     zeros_index = np.where(data[:, 1] == 0)[0]
#     if len(zeros_index) != 0:
#         vimps[zeros_index] = 0
#
#     if show:
#         plt.plot(np.arange(data.shape[0]), vimps, '.k')
#         sorted_index = np.argsort(-vimps)  # 反置，则越大越重要，argsort（a,ax）将矩阵a按照ax从小到大排序，返回排序后的下标
#         for i in range(15):
#             plt.text(sorted_index[i], vimps[sorted_index[i]], str(sorted_index[i]))
#         plt.title(file_dir[36:])
#         plt.xlabel('feature idx')
#         plt.ylabel('feature score')
#         # plt.ylim([0,0.25])
#         plt.show()
#         # print(sorted_index[:20])
#     return vimps
# def load_feature_impts_from_dir(Data_path, file_dir=r'save_vimps', show=False, top_num_genes=20):
#     '''
#         导入基因重要性特征
#     '''
#     file_list = os.listdir(file_dir)
#
#     feature_list = pd.read_csv(Data_path).iloc[:,1:]
#     all_features_name_list = list(feature_list)
#     all_features_name_index = {}  ##作用是通过列名找到对应的索引----列表和索引一一对应
#     all_features_index_name = {}  ##作用是通过索引找到对应的列名----列表和索引一一对应
#     for i, tmp_feat_name in enumerate(all_features_name_list):
#         all_features_name_index[tmp_feat_name] = i
#         all_features_index_name[i] = tmp_feat_name
#
#     for i, file in enumerate(file_list[:]):
#         file_path = os.path.join(file_dir, file)
#         if i == 0:
#             data = np.load(file_path, allow_pickle=True)
#         else:
#             data += np.load(file_path, allow_pickle=True)
#
#     vimps = data[:, 0] / data[:, 1]
#
#     non_zero_count = np.count_nonzero(data[:, 1])
#     print("被抽到特征的数量:", non_zero_count)
#
#     if show == True:
#         plt.figure(figsize=(16, 8))
#         plt.plot(np.arange(data.shape[0]), data[:, 1], '.k')
#         plt.xlabel('Feature Index')
#         plt.ylabel('Checked feature Count')
#         plt.title('Feature Count')
#         plt.show()
#
#         plt.plot(np.arange(data.shape[0]), vimps, '.k')
#         sorted_index = np.argsort(-vimps)
#
#         # 标注前 top_num_genes 个基因，包括名称和得分
#         for i in range(top_num_genes):
#             gene_index = sorted_index[i]
#             gene_name = all_features_name_list[gene_index]
#             gene_score = vimps[gene_index]
#
#             print(f"Gene {gene_name}, Index {gene_index}, Score {gene_score}")
#
#             plt.text(gene_index, vimps[gene_index], str(gene_name))
#
#         plt.title('Gene Importance')
#         plt.xlabel('Feature Index')
#         plt.ylabel('Feature Score')
#         plt.show()
#
#     return vimps

# def load_feature_impts_from_dir(Data_path, file_dir=r'save_vimps', show=False, top_num_genes=20):
#     '''
#         导入基因重要性特征
#     '''
#     file_list = os.listdir(file_dir)
#
#     feature_list = pd.read_csv(Data_path).iloc[:, 1:]
#     all_features_name_list = list(feature_list)
#     all_features_name_index = {}  # 作用是通过列名找到对应的索引----列表和索引一一对应
#     all_features_index_name = {}  # 作用是通过索引找到对应的列名----列表和索引一一对应
#     for i, tmp_feat_name in enumerate(all_features_name_list):
#         all_features_name_index[tmp_feat_name] = i
#         all_features_index_name[i] = tmp_feat_name
#
#     for i, file in enumerate(file_list[:]):
#         file_path = os.path.join(file_dir, file)
#         if i == 0:
#             data = np.load(file_path, allow_pickle=True)
#         else:
#             data += np.load(file_path, allow_pickle=True)
#
#     vimps = data[:, 0] / data[:, 1]
#
#     non_zero_count = np.count_nonzero(data[:, 1])
#     print("被抽到特征的数量:", non_zero_count)
#
#     if show:
#         plt.figure(figsize=(16, 8))
#         plt.plot(np.arange(data.shape[0]), data[:, 1], '.k')
#         plt.xlabel('Feature Index')
#         plt.ylabel('Checked feature Count')
#         plt.title('Feature Count')
#         plt.show()
#
#         plt.figure(figsize=(16, 8))
#         sorted_index = np.argsort(-vimps)
#
#         # 为前 top_num_genes 个基因设置散点颜色和标注颜色
#         for i in range(len(vimps)):
#             if i < 16:  # 前16个特征
#                 color = 'red'  # 红色
#             else:
#                 color = 'black'  # 其他特征为黑色
#
#             plt.scatter(i, vimps[i], color=color)
#
#             # 标注基因名称
#             if i < 16:  # 仅为前16个特征添加标注
#                 plt.text(i, vimps[i], str(all_features_name_list[i]), color=color)
#
#         # 标注前 top_num_genes 个基因，包括名称和得分
#         for i in range(top_num_genes):
#             gene_index = sorted_index[i]
#             gene_name = all_features_name_list[gene_index]
#             gene_score = vimps[gene_index]
#
#             print(f"Gene {gene_name}, Index {gene_index}, Score {gene_score}")
#
#             plt.text(gene_index, vimps[gene_index], str(gene_name), color='red')
#
#         plt.title('Gene Importance')
#         plt.xlabel('Feature Index')
#         plt.ylabel('Feature Score')
#         plt.show()
#
#     return vimps
# def load_feature_impts_from_dir(Data_path, file_dir=r'save_vimps', show=False, top_num_genes=20):
#     '''
#         导入基因重要性特征
#     '''
#     file_list = os.listdir(file_dir)
#
#     feature_list = pd.read_csv(Data_path).iloc[:, 1:]
#     all_features_name_list = list(feature_list)
#
#     for i, file in enumerate(file_list[:]):
#         file_path = os.path.join(file_dir, file)
#         if i == 0:
#             data = np.load(file_path, allow_pickle=True)
#         else:
#             data += np.load(file_path, allow_pickle=True)
#
#     vimps = data[:, 0] / data[:, 1]
#     non_zero_count = np.count_nonzero(data[:, 1])
#     print("被抽到特征的数量:", non_zero_count)
#
#     if show:
#         plt.figure(figsize=(16, 8))
#
#         # 绘制 Checked feature Count
#         plt.plot(np.arange(data.shape[0]), data[:, 1], '.k')
#         plt.xlabel('Feature Index')
#         plt.ylabel('Checked feature Count')
#         plt.title('Feature Count')
#         plt.show()
#         #
#         # plt.figure(figsize=(16, 8))
#         sorted_index = np.argsort(-vimps)
#
#         # 绘制散点图
#         plt.plot(np.arange(data.shape[0]), vimps, '.k')
#
#         # 仅标注得分最高的前16个基因
#         for i in range(6):
#             gene_index = sorted_index[i]
#             gene_name = all_features_name_list[gene_index]
#             gene_score = vimps[gene_index]
#
#             # 标记颜色为红色
#             plt.text(gene_index, gene_score, str(gene_name), color='red')
#
#         # 标注前 top_num_genes 个基因，包括名称和得分
#         for i in range(top_num_genes):
#             gene_index = sorted_index[i]
#             gene_name = all_features_name_list[gene_index]
#             gene_score = vimps[gene_index]
#             print(f"Gene {gene_name}, Index {gene_index}, Score {gene_score}")
#
#         plt.title('Gene Importance')
#         plt.xlabel('Feature Index')
#         plt.ylabel('Feature Score')
#         plt.show()
#
#     return vimps

def load_feature_impts_from_dir(Data_path, file_dir=r'save_vimps', show=False, top_num_genes=20):
    '''
        导入基因重要性特征
    '''
    file_list = os.listdir(file_dir)

    feature_list = pd.read_csv(Data_path).iloc[:, 1:]
    all_features_name_list = list(feature_list)

    for i, file in enumerate(file_list[:]):
        file_path = os.path.join(file_dir, file)
        if i == 0:
            data = np.load(file_path, allow_pickle=True)
        else:
            data += np.load(file_path, allow_pickle=True)

    vimps = data[:, 0] / data[:, 1]
    non_zero_count = np.count_nonzero(data[:, 1])
    print("被抽到特征的数量:", non_zero_count)

    if show:
        plt.figure(figsize=(16, 8))

        # 绘制 Checked feature Count
        plt.plot(np.arange(data.shape[0]), data[:, 1], '.k')
        plt.xlabel('Feature Index')
        plt.ylabel('Checked feature Count')
        plt.title('Feature Count')
        plt.show()

        # plt.figure(figsize=(16, 8))
        sorted_index = np.argsort(-vimps)

        # 绘制散点图
        plt.plot(np.arange(data.shape[0]), vimps, '.k')

        # 仅标注得分最高的前16个基因
        for i in range(6):
            gene_index = sorted_index[i]
            gene_name = all_features_name_list[gene_index]
            gene_score = vimps[gene_index]

            # 标记颜色为红色
            if i == 2:  # 第五个特征
                plt.text(gene_index, gene_score, str(gene_name), color='red', ha='left', va='bottom', fontsize=8)
            elif i == 3:  # 第六个特征
                plt.text(gene_index, gene_score + 0.005, str(gene_name), color='red', ha='left', va='bottom',
                         fontsize=8)  # 向上偏移
            else :

                plt.text(gene_index, gene_score, str(gene_name), color='red', ha='left', va='bottom', fontsize=8)

        # 标注前 top_num_genes 个基因，包括名称和得分
        for i in range(top_num_genes):
            gene_index = sorted_index[i]
            gene_name = all_features_name_list[gene_index]
            gene_score = vimps[gene_index]
            print(f"Gene {gene_name}, Index {gene_index}, Score {gene_score}")

        plt.title('Gene Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Score')
        plt.show()

    return vimps


def load_and_show_feature_std(vimps, file_dir=r'save_vimps'):
    file_list = os.listdir(file_dir)
    for i, file in enumerate(file_list):
        file_path = os.path.join(file_dir, file)
        if i == 0:
            t = np.load(file_path)
            data = np.zeros([len(file_list), t.shape[0]])
            data[i, :] = t[:, 0] / t[:, 1]
        else:
            t = np.load(file_path)
            data[i, :] = t[:, 0] / t[:, 1]

    std = data.std(axis=0)
    plt.plot(data.std(axis=0), '.k')
    plt.title('std')
    plt.show()

    index = np.where(vimps > 0.005)[0]
    index = np.where(std > 0.01)[0]
    plt.boxplot(data[:, index], showmeans=True)

    plt.xticks(range(1, len(index) + 1), index)
    plt.show()

    # for i in np.where(std > 0.02)[0]:
    #     plt.plot(data[:, i],'.', label=str(i))
    # plt.legend()
    # plt.plot(data[:, 14],'.k')
    # plt.show()
    return std


class Vimps:
    """
        计算特征重要性时，用来记录每次抽取的特征重要性。
        一个简单的数据结构
    """
    def __init__(self, num_of_features, save_dir=None):
        self.vimps = np.zeros(num_of_features, )
        self.count_feature = np.zeros(num_of_features, )
        self.save_dir = save_dir
        self.__check()

    def __check(self):
        if self.save_dir is not None and not os.path.isdir(self.save_dir):
            raise Exception(f'This dir {self.save_dir} is not exist.')

    def update(self, n_feature_vimps, feature_count=None):
        if isinstance(n_feature_vimps, dict) and isinstance(feature_count, dict):
            for key, value in n_feature_vimps.items():
                self.vimps[key] += value
                self.count_feature[key] += feature_count[key]

        elif isinstance(n_feature_vimps, np.ndarray) and isinstance(feature_count, np.ndarray):
            self.vimps += n_feature_vimps
            self.count_feature += feature_count

    def get_avg_vimps(self):
        not_nan_idx = np.where(self.count_feature != 0)
        return self.vimps[not_nan_idx] / self.count_feature[not_nan_idx]

    def save_vimps(self, save_path=None):
        if self.save_dir is not None and save_path is None:
            save_path = os.path.join(self.save_dir, f'vimps_and_count_{time.time()}.npy')
        elif self.save_dir is None and save_path is None:
            save_path = f'vimps_and_count_{time.time()}.npy'
        elif self.save_dir is not None and save_path is not None:
            save_path = os.path.join(self.save_dir, save_path)
        np.save(save_path, np.hstack([self.vimps.reshape(-1, 1), self.count_feature.reshape(-1, 1)]))
