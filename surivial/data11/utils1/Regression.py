import random
import time
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
import warnings

from surivial.data11.utils1.feature_important_tools import Vimps

warnings.filterwarnings("ignore", category=FutureWarning)
class Regression:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.best_regr = None
        self.ensembles = self.build_ensemble()

    def build_ensemble(self):
        """
        创建集成回归模型
        """
        gpr = GaussianProcessRegressor()
        kg = KNeighborsRegressor()
        dt = DecisionTreeRegressor()
        rf = RandomForestRegressor(n_estimators=200)
        ada = AdaBoostRegressor()

        lg = LinearRegression()
        gbr = GradientBoostingRegressor()
        brg = BaggingRegressor()
        xgb = XGBRegressor()
        lgbm = LGBMRegressor(verbose=-1)
        cat = CatBoostRegressor(logging_level='Silent')



        self.ensembles = [gbr,rf,dt,brg,xgb,lgbm,ada,gpr,cat,kg,lg]
        # self.ensembles = [ lg]


        return self.ensembles

    def fit(self, x_train, y_train):
        """
        fit every regression in ensembles
        """
        self.ensembles = [regr.fit(x_train, y_train) for regr in self.ensembles]
        return self

    def find_best_ensembles(self, x_test, y_test, R2=False):
        """
        训练所有的分类器:ensemble_fitted，使用测试集合得出每个分类器的准确性
        """
        if R2:
            scores = [regr.score(x_test, y_test) for regr in self.ensembles]
        else:
            scores = [0 for _ in self.ensembles]
            for i, regr in enumerate(self.ensembles):
                try:
                    y_pred = regr.predict(x_test)
                    scores[i] = self.model_evaluate(y_true=y_test, y_pred=y_pred)['r2']
                except NotFittedError as e:
                    scores[i] = 0

        idx = np.argsort(scores)[-1]
        self.best_regr = self.ensembles[idx]
        self.best_score = scores[idx]
        return idx, self.best_regr, self.best_score

    def predict_by_voted(self, x_test):
        """
        每一种模型进行预测，预测结果取平均值
        """
        y_preds = [clf.predict(x_test) for clf in self.ensembles]
        y_preds = np.array(y_preds).reshape(len(y_preds), -1)
        y_pred = np.mean(y_preds, axis=0)
        return y_pred

    def predict_by_best_clf(self, x_test):
        """
        使用最好的分类器进行预测
        """
        y_pred = self.best_regr.predict(x_test)
        return y_pred

    def predict(self, x_test, vote=False):
        """
        两种预测方法:
        一: 使用投票策略，使用所有的分类器进行预测，将所有的lable进行投票，投票结果为最终结果
        二: 使用最好的分类器，进行预测，预测结果为最终结果
        """
        if vote:
            y_pred = self.predict_by_voted(x_test)
        else:
            if self.best_regr is None:
                raise ValueError('The best_clf is not defined!\n You should execute the function named "find_best_ensembles()"!')
            y_pred = self.predict_by_best_clf(x_test)
        return y_pred

    def get_best_score(self, x_test, y_test, R2=True, vote=False):
        """
        得到最好的score，也是有两种方式
        一: 使用投票/最好分类器 来得到预测结果
        二: 使用决策系数(R^2)或者平均绝对误差(MAE)来作为最后的结果
        """
        y_pred = self.predict(x_test, vote=vote)
        if R2:
            self.best_score = self.model_evaluate(y_true=y_test, y_pred=y_pred)['r2']
        else:
            self.best_score = self.model_evaluate(y_true=y_test, y_pred=y_pred)['mae']
        return self.best_score

    @staticmethod
    def model_evaluate(y_true, y_pred):
        """
        衡量线性回归的MSE、RMSE、MAE、R2
        """
        mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
        rmse = np.sqrt(mse)
        mae = np.sum(np.absolute(y_true - y_pred)) / len(y_true)
        r2 = 1 - mse / np.var(y_true)  # 均方误差/方差
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def get_shuffled_data(data, dim, random_state=None):
    """
    打乱数据集的顺序，并返回
    """
    myrandom = np.random.RandomState(random_state)
    index = np.arange(data.shape[0])
    myrandom.shuffle(index)
    t_data = data.copy()
    t_data.iloc[:, dim] = data.iloc[index, dim].values
    return t_data


def train_and_trainscore(X, y):
    """
    训练模型并返回最佳得分、特征重要性和最佳分类器类型
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    regression = Regression()
    regression.fit(x_train, y_train)
    _, best_clf, best_score = regression.find_best_ensembles(x_test, y_test)
    best_clf_type = type(best_clf).__name__
    vimps = np.zeros((x_test.shape[1]))
    for n_dim in range(x_test.shape[1]):
        shuffle_one_feature_data = get_shuffled_data(x_test, n_dim)
        score_shuffled = regression.get_best_score(shuffle_one_feature_data, y_test,  R2=True,vote=False)
        one_feature_imptns = abs(best_score - score_shuffled)
        vimps[n_dim] = one_feature_imptns
    return best_score, vimps, best_clf_type

def extract_feature_for_acc(X, y, n_features, times, save_dir):

    classifier_counts = {}
    best_score_accuracy = []

    sample = range(X.shape[1])
    vimps_ans = Vimps(X.shape[1], save_dir)
    feature_dict = {}
    feature_count = {}
    for j in tqdm(range(times)):
        extract_features = random.sample(sample, n_features)
        extract_features = np.array(extract_features)
        best_score, vimps, best_clf_type = train_and_trainscore(X.iloc[:, extract_features], y)

        best_score = np.round(best_score,2)
        for i, vimp in zip(extract_features, vimps):
            if i in feature_dict:
                feature_dict[i] += vimp
                feature_count[i] += 1
            else:
                feature_dict[i] = vimp
                feature_count[i] = 1
        if best_clf_type in classifier_counts:
            classifier_counts[best_clf_type] += 1
        else:
            classifier_counts[best_clf_type] = 1
        best_score_accuracy.append(best_score)

    vimps_ans.update(feature_dict, feature_count)
    vimps_ans.save_vimps()
    print('End:', time.ctime())
    return best_score_accuracy, classifier_counts

