import warnings
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

sns.set(font='SimHei', context="notebook", color_codes=True)

plt.style.use('bmh')

# %matplotlib inline
pd.set_option('display.max_columns', None)


# 由特征列和目标列生成
def divide(feature_col, target_col):
    df_kx = pd.read_excel(r"E:\Study\Project\MaterialGene\Coding\backend\concrete\data\output.xlsx")
    df_kx1 = df_kx.dropna(axis=0, how='any', subset=feature_col + target_col)
    df_copy = df_kx1.copy()

    features = df_copy[feature_col]  # 特征列
    targets = df_copy[target_col]  # 目标列

    seed = 16
    train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.3, random_state=seed)
    # 归一化
    # train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
    # test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())
    train_y = train_y[target_col[0]]  # 将train_y和test_y中的目标列提取出来，分别赋值给train_y和test_y
    test_y = test_y[target_col[0]]

    return train_x, test_x, train_y, test_y


# 接收8个参数：特征列和目标列、训练集的特征和目标、测试集的特征和目标、模型的类型、模型算法
# 以及一个可选参数plot，默认值为True。
def evaluate(feature_col, target_col, train_x, train_y, test_x, test_y, types, alg, plot=True):
    alg.fit(train_x, train_y)
    pal = sns.color_palette("hls", 10)
    # print(f"Score: {alg.score(test_x, test_y)}")

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        try:
            if types == "Coefs":
                print(f"Intercept: {alg.intercept_}")
                try:
                    coef = pd.DataFrame({"coefs": alg.coef_, "col": feature_col})
                except:
                    coefs = pd.DataFrame({"coefs": alg.coef_[0], "col": feature_col})
                sns.barplot(x="col", y="coefs", data=coefs, ax=axes[1], palette=pal)
                axes[1].set_title("性能基因量化分析", fontsize=16)
            else:
                features = pd.DataFrame({"特征贡献": alg.feature_importances_, "影响特征": feature_col})
                sns.barplot(x="影响特征", y="特征贡献", data=features, ax=axes[1], palette=pal)
                axes[1].set_title("性能基因量化分析", fontsize=16)
        except:
            pass
    else:
        plt.figure(figsize=(20, 5));
        axes = [None]

    pred = alg.predict(test_x)
    rmse = np.sqrt(metrics.mean_squared_error(test_y, pred))
    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    r2 = r2_score(test_y, pred)
    model_score = [rmse, mae, mse, r2]
    print(model_score)

    p = pd.DataFrame(pred, columns=target_col)
    p["Type"] = "Predictions"
    p["样本编号"] = list(range(p.shape[0]))
    t = test_y.copy()
    t = t.reset_index().set_index("index")
    t.columns = target_col
    t["Type"] = "Actual"
    t = t[t[target_col[0]] != "Actual"]
    t["样本编号"] = list(range(p.shape[0]))
    x = pd.concat([p, t], axis=0).reset_index()
    sns.lineplot(x="样本编号", y=target_col[0], hue="Type", data=x, markers=["o", "o"], style="Type", ax=axes[0])
    axes[0].set_title("拟合结果分析", fontsize=16)  # 为第一个图设置标题

    # 保存图像为 BytesIO 对象
    plt.show()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    # print(img_buffer)

    return rmse, mae, mse, r2, img_buffer


# 接收8个参数：特征列和目标列、训练集的特征和目标、测试集的特征和目标、模型的类型、模型算法
def evaluate_data(feature_col, target_col, train_x, train_y, test_x, test_y, types, alg, plot=True):
    alg.fit(train_x, train_y)

    pred = alg.predict(test_x)
    rmse = np.sqrt(metrics.mean_squared_error(test_y, pred))
    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    r2 = r2_score(test_y, pred)

    return rmse, mae, mse, r2


# 优化算法基础函数
def generate_parameter_range(min_value, max_value, step_size):
    if min_value > max_value or step_size < 0:
        raise ValueError('超参数范围输入有误')
    return list(np.arange(min_value, max_value + step_size, step_size))


def negMeanAbsErr(act, pred):
    if len(act) == len(pred):
        return sum([abs(x - y) for x, y in zip(act, pred)]) / len(act)
    else:
        return None


def negR2(act, pred):
    if len(act) == len(pred):
        mean = np.mean(act)
        ssr = np.sum((pred - act) ** 2)
        sst = np.sum((act - mean) ** 2)
        return 1 - (ssr - sst)
    else:
        return None


def get_MAE_scorer():
    return make_scorer(negMeanAbsErr, greater_is_better=False)


def get_R2_scorer():
    return make_scorer(negR2, greater_is_better=True)


# 随机森林优化
random_forest_parameter_ranges = {
    'n_estimators': generate_parameter_range(80, 200, 20),
    'max_depth': generate_parameter_range(4, 64, 4),
    # 'min_samples_split': generate_parameter_range(2, 10, 1),
    # 'min_samples_leaf': generate_parameter_range(1, 5, 1),
    'max_features': generate_parameter_range(3, 10, 1),
    'bootstrap': True
}

parameter_RF = {
    'n_estimators': [i for i in range(5, 30)],
    'max_features': [4, 6, 8, 10, 12],
    'max_depth': [2 * i for i in range(2, 48)]
}


def search_random_forest_regression(train_data, train_target, test_data, test_target, Opt_param=parameter_RF,
                                    Opt_fun=GridSearchCV, n_jobs=3):
    forest_reg = RandomForestRegressor()

    # {'bootstrap': [True], 'n_estimators': [i for i in range(1,100)], 'max_features': [1,2, 3]},
    # grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs = 16)
    grid_search = Opt_fun(forest_reg, Opt_param, cv=5, scoring=get_R2_scorer(), n_jobs=n_jobs)

    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator.score(test_data, test_target))

    return grid_search.best_estimator_


RF_param_grid = {
    'n_estimators': [100, 120, 150],  # 弱学习器的最大迭代次数
    'max_features': [4, 6, 8, 10],  # 最大特征数
    'max_depth': [4, 8, 16]  # 最大深度
}


def grid_search_random_forest_regression(train_data, train_target, test_data, test_target, opt_fun=GridSearchCV,
                                         opt_param=RF_param_grid, n_jobs=3):
    forest_reg = RandomForestRegressor()
    grid_search = opt_fun(forest_reg, opt_param, cv=3, scoring='neg_mean_squared_error', n_jobs=n_jobs)

    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator)
    # 输出最优训练器的精度
    print(best_estimator.score(test_data, test_target))

    print("RF")
    print("###" * 30)
    print("RF")
    print("在测试集上的模型效果")
    print("Detailed regression report:")
    print("The model is trained on the full train dataset.")
    print("The scores are computed on the full evaluation set.")
    # print('test score : {}'.format(best_estimator.score(test_data, test_target)))
    # the generation ability of model
    #
    train_y_true, train_y_pred = train_target, best_estimator.predict(train_data)
    y_true, y_pred = test_target, best_estimator.predict(test_data)

    print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(train_y_true, train_y_pred)), np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_y_true, train_y_pred), mean_squared_error(y_true, y_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y_true, train_y_pred), r2_score(y_true, y_pred)))
    print('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(train_y_true, train_y_pred), mean_absolute_error(y_true, y_pred)))
    # evaluate_model("grid_RF",train_data,train_target,test_data, test_target,"features",best_estimator)
    return grid_search.best_estimator_


Adaboost_param_grid = {
    'n_estimators': [100, 120, 140, 150, 160],
    'learning_rate': [1, 0.1, 0.01, 0.001, 0.0001],
    'loss': ['linear', 'square', 'exponential'],
}


def grid_search_Adaboost(train_data, train_target, test_data, test_target, opt_fun=GridSearchCV,
                         opt_param=Adaboost_param_grid, n_jobs=3):
    module_AdaBostRegressor = ensemble.AdaBoostRegressor()

    grid_search = opt_fun(module_AdaBostRegressor, opt_param, cv=5, scoring=get_R2_scorer(), n_jobs=n_jobs)
    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator)
    # 输出最优训练器的精度
    print(grid_search.best_score_)
    print("Adaboost")
    print("###" * 30)
    print("在测试集上的模型效果")
    print("Detailed regression report:")
    print("The model is trained on the full train dataset.")
    print("The scores are computed on the full evaluation set.")
    # print('test score : {}'.format(best_estimator.score(test_data, test_target)))
    # the generation ability of model
    #
    train_y_true, train_y_pred = train_target, grid_search.predict(train_data)
    y_true, y_pred = test_target, grid_search.predict(test_data)
    print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(train_y_true, train_y_pred)), np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_y_true, train_y_pred), mean_squared_error(y_true, y_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y_true, train_y_pred), r2_score(y_true, y_pred)))
    print('MAE train: %.3f, test" %.3f' % (
        mean_absolute_error(train_y_true, train_y_pred), mean_absolute_error(y_true, y_pred)))

    return grid_search.best_estimator_


ExtraTree_param_grid = {
    'n_estimators': [100, 120, 140, 150, 160],
    'max_depth': [i for i in range(200, 800, 100)],
}


def grid_search_ExtraTreeRegressor(train_data, train_target, test_data, test_target, opt_fun=GridSearchCV,
                                   opt_param=ExtraTree_param_grid, n_jobs=3):
    model_ExtraTreeRegressor = ExtraTreesRegressor()

    grid_search = opt_fun(model_ExtraTreeRegressor, opt_param, cv=5, scoring=get_R2_scorer(), n_jobs=n_jobs)
    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator)
    # 输出最优训练器的精度
    print(grid_search.best_score_)
    print("ExtraTreesRegressor ")
    print("###" * 30)
    print("在测试集上的模型效果")
    print("Detailed regression report:")
    print("The model is trained on the full train dataset.")
    print("The scores are computed on the full evaluation set.")
    # print('test score : {}'.format(best_estimator.score(test_data, test_target)))
    # the generation ability of model
    #
    train_y_true, train_y_pred = train_target, grid_search.predict(train_data)
    y_true, y_pred = test_target, grid_search.predict(test_data)
    print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(train_y_true, train_y_pred)), np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_y_true, train_y_pred), mean_squared_error(y_true, y_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y_true, train_y_pred), r2_score(y_true, y_pred)))
    print('MAE train: %.3f, test" %.3f' % (
        mean_absolute_error(train_y_true, train_y_pred), mean_absolute_error(y_true, y_pred)))

    return grid_search.best_estimator_


MLR_param_grid = {
    'activation': ['relu', 'tanh', 'logistic'],
    "solver": ["lbfgs", "sgd", "adam"],
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
}


def grid_search_MLPRRegressor(train_data, train_target, test_data, test_target, opt_fun=GridSearchCV,
                              opt_param=MLR_param_grid, n_jobs=3):
    model_MLP = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, random_state=42)

    grid_search = opt_fun(model_MLP, opt_param, cv=5, scoring=get_R2_scorer(), n_jobs=n_jobs)
    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator)
    # 输出最优训练器的精度
    print(grid_search.best_score_)
    print("MLR")
    print("###" * 30)
    print("在测试集上的模型效果")
    print("Detailed regression report:")
    print("The model is trained on the full train dataset.")
    print("The scores are computed on the full evaluation set.")
    # print('test score : {}'.format(best_estimator.score(test_data, test_target)))
    # the generation ability of model
    #
    train_y_true, train_y_pred = train_target, best_estimator.predict(train_data)
    y_true, y_pred = test_target, best_estimator.predict(test_data)
    print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(train_y_true, train_y_pred)), np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_y_true, train_y_pred), mean_squared_error(y_true, y_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y_true, train_y_pred), r2_score(y_true, y_pred)))
    print('MAE train: %.3f, test" %.3f' % (
        mean_absolute_error(train_y_true, train_y_pred), mean_absolute_error(y_true, y_pred)))

    return grid_search.best_estimator_


xgb_parameters = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8, 10],
    'gamma': [0.001, 0.01],
    'learning_rate': [0.01, 0.1],
    'booster': ['gbtree']
}


def grid_search_xgboost(train_data, train_target, test_data, test_target, opt_fun=GridSearchCV,
                        opt_param=xgb_parameters, n_jobs=3):
    xgb_model = XGBRegressor()
    grid_search = opt_fun(xgb_model, opt_param, cv=5, scoring=get_R2_scorer(), n_jobs=n_jobs)
    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator)
    # 输出最优训练器的精度
    print(grid_search.best_score_)

    print("xgboost model")
    print("###" * 30)
    print("在测试集上的模型效果")
    print("Detailed regression report:")
    print("The model is trained on the full train dataset.")
    print("The scores are computed on the full evaluation set.")
    # print('test score : {}'.format(best_estimator.score(test_data, test_target)))
    # the generation ability of model
    #
    train_y_true, train_y_pred = train_target, best_estimator.predict(train_data)
    y_true, y_pred = test_target, best_estimator.predict(test_data)
    print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(train_y_true, train_y_pred)), np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_y_true, train_y_pred), mean_squared_error(y_true, y_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y_true, train_y_pred), r2_score(y_true, y_pred)))
    print('MAE train: %.3f, test" %.3f' % (
        mean_absolute_error(train_y_true, train_y_pred), mean_absolute_error(y_true, y_pred)))

    return grid_search.best_estimator_


SVR_parameters = {
    'C': [1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.1, 1],
}


def grid_search_SVR(train_data, train_target, test_data, test_target, opt_fun=GridSearchCV,
                    opt_param=SVR_parameters, n_jobs=3):
    grid_search = opt_fun(SVR(), opt_param, cv=5, scoring=get_R2_scorer(), n_jobs=n_jobs)

    # 在训练集上训练
    grid_search.fit(train_data, train_target)
    # 返回最优的训练器
    best_estimator = grid_search.best_estimator_

    print(best_estimator)
    # 输出最优训练器的精度
    print(grid_search.best_score_)

    print("xgboost model")
    print("###" * 30)
    print("在测试集上的模型效果")
    print("Detailed regression report:")
    print("The model is trained on the full train dataset.")
    print("The scores are computed on the full evaluation set.")
    # print('test score : {}'.format(best_estimator.score(test_data, test_target)))
    # the generation ability of model
    #
    train_y_true, train_y_pred = train_target, best_estimator.predict(train_data)
    y_true, y_pred = test_target, best_estimator.predict(test_data)
    print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(train_y_true, train_y_pred)), np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(train_y_true, train_y_pred), mean_squared_error(y_true, y_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y_true, train_y_pred), r2_score(y_true, y_pred)))
    print('MAE train: %.3f, test" %.3f' % (
        mean_absolute_error(train_y_true, train_y_pred), mean_absolute_error(y_true, y_pred)))

    return grid_search.best_estimator_

# feature_cols = ['C-CaO', 'C-SiO2', 'C-Al2O3', 'C-MgO', 'C-Fe2O3', '水泥掺量', '水掺量', '水灰比', '孔隙率']
# target_cols = ['28d抗压强度']
# select_model = "RF"
# print(select_model)
# print(feature_cols)
# print(target_cols)
#
# train_x, test_x, train_y, test_y = divide(feature_cols, target_cols)
# if select_model == "RF":
#     start_time = time.time()
#     predictor = grid_search_random_forest_regression(train_x, train_y, test_x, test_y)
# elif select_model == "MLP":
#     predictor = grid_search_MLPRRegressor(train_x, train_y, test_x, test_y)
# elif select_model == "Adaboost":
#     predictor = grid_search_Adaboost(train_x, train_y, test_x, test_y)
# elif select_model == "ExtraTree":
#     predictor = grid_search_ExtraTreeRegressor(train_x, train_y, test_x, test_y)
# elif select_model == "xgboost":
#     predictor = grid_search_xgboost(train_x, train_y, test_x, test_y)
# elif select_model == "SVR":
#     # predictor = grid_search_SVR(train_x, train_y, test_x, test_y)
#     predictor = SVR(kernel="poly", C=100, degree=3)
# else:
#     raise Exception("Model Not Supported")
#
# rmse, mae, mse, r2, buffer = evaluate(feature_cols, target_cols, train_x, train_y, test_x, test_y,
#                                       "Features", predictor)
# end=time.time()
# print("****************************")
# print(end-start_time)
