import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data11 import Regression
from data11 import get_gaussian_boundary
from data11 import load_feature_impts_from_dir

if __name__ == '__main__':
    # 设置数据名称
    name = 'GBM'
    # 设置数据存取路径
    WORK_PATH = r'..'
    # Data_path = os.path.join(WORK_PATH, f'output/clean/TCGA-{name}.csv')
    # vimps_save_dir = os.path.join(WORK_PATH, f'output/vimps/vimps_{name}/vimps_for_all_RNA_10feature-3')
    # vimps_GMM_save_dir = os.path.join(WORK_PATH, f'output/vimps/vimps_GMM_{name}')

    Data_path = os.path.join(WORK_PATH, f'E:\desktop\surivial\data\merged_mirna2.csv')
    vimps_save_dir = os.path.join(WORK_PATH, f'output/vimps/vimps_{name}/vimps_for_all_cnv_10feature-1')
    vimps_GMM_save_dir = os.path.join(WORK_PATH, f'output/vimps/vimps_GMM_{name}_cnv')
    # 创建文件夹
    if not os.path.exists(vimps_save_dir):
        os.makedirs(vimps_save_dir)

    if not os.path.exists(vimps_GMM_save_dir):
        os.makedirs(vimps_GMM_save_dir)
    # 按名称提取数据
    data = pd.read_csv(Data_path, index_col=0)
    X = data.loc[data.vital_status==1, ~data.columns.isin(['days_to_death'])]
    y = data.loc[data.vital_status==1, 'days_to_death']


    #################### 多进程抽取特征 ####################
    # works = 100
    # 设置每次抽取的特征数
    n_feature = 10
    # # rounds数
    # times_for_a_work = int(data.shape[1] * (1000 / n_feature) / works)
    # # 设置线程数++
    # if os.cpu_count() == 16:
    #     n_jobs = 10
    # elif os.cpu_count() == 12:
    #     n_jobs = 10
    # else:
    #     n_jobs = int(os.cpu_count() * 0.8 // 2 * 2)
    # jobs = []
    times_for_a_work = 10
    # 设置score的阈值，这里是用MAE
    score_cut = 10 # 十年

    best_score_accuracy, regressor_counts = Regression.extract_feature_for_acc(X, y, n_feature, times_for_a_work, vimps_save_dir, score_cut)





    # 将计算好的特征重要性值取出来
    vimps = load_feature_impts_from_dir(vimps_save_dir, show=True)

    # 拟合高斯来判断应该取多少特征（最后一个边界）
    vimps_gau = get_gaussian_boundary(vimps, n_components=20, show=True)

    # 最大边界的值为：vimps_max
    vimps_max = vimps_gau[len(vimps_gau) - 1]
    print("最大边界为：", vimps_max)
    selected_name = []
    selected_id = []
    selected_vimp = []
    count = 0
    # 大于（等于）最大边界的特征则使用
    for i, vimp in enumerate(vimps):  # 此时的vimps已经是[2:],去了生存时间和生存状态
        # if vimp >= vimps_max:
            count = count + 1
            selected_name.append(X.columns[i])  # i+2
            selected_id.append(i)
            selected_vimp.append(vimp)
            selectedlists = {"name": selected_name, "id": selected_id, "vimp": selected_vimp}  # 候选基因保存在selectedlists中
            print(i, X.columns[i], sep=':')
    print(count)

    # 特征重要性得分排序(降序)
    decrease_sort = np.argsort(-np.array(selectedlists['vimp']))
    # print([vimps_GMM['vimp'][i] for i in decrease_sort])    ##查看降序后的特征得分
    selectedlists['name'] = [selectedlists['name'][i] for i in decrease_sort]
    selectedlists['id'] = [selectedlists['id'][i] for i in decrease_sort]
    selectedlists['vimp'] = [selectedlists['vimp'][i] for i in decrease_sort]

    # 保存GMM选出来的重要特征
    json_file = open(os.path.join(vimps_GMM_save_dir, 'vimps_GMM.json'), 'w')
    json_file.write(json.dumps(selectedlists))
    json_file.close()