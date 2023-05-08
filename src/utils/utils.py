import functools
import math
import operator

import numpy as np
import torch
from torch.utils.data import DataLoader


def saveModel(pkl_filename, obj):
    """
    持久化对象

    Parameters
    ----------
        pkl_filename : string
            需要保存的对象的路径。默认保存位置在当前文件夹下，要保存到其他地方，需要把路径写完整。
            文件类型为.pkl

        obj : object
            需要保存的对象

    """
    import joblib
    joblib.dump(filename=pkl_filename, value=obj)


def loadModel(pkl_filename):
    """
    读取对象

    Parameters
    ----------
        pkl_filename :
            需要读取的对象的路径。默认读取位置在当前文件夹下，要读取其他地方的对象，需要把路径写完整。
            文件类型为.pkl

    Returns
    -------
        obj : object
            需要读取的对象

    """
    import joblib
    obj = joblib.load(pkl_filename)
    return obj


def findBestParmByGridSearchCv(X, y, estimator, parm_grid, cv_num=5, n_jobs=-1):
    """
    网格搜索寻找最优参数

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        estimator :数据类型为'object'
            基学习器对象。
            该对象被认为是实现了sklearn estimator接口的对象。按需求直接使用sklearn中的类对象。
            例如需要使用knn回归来作为基学习器，就必须要导入KNeighborsRegressor这个类，并且将该类的对象赋值给estimator参数,base_estimator=KNeighborsRegressor()。
            具体学习器对象请看下面Notes

        parm_grid : 数据类型为'字典'
            待优化参数
            key为需要优化的参数名。value为列表。
            例如parm_grid = {"C": [0.1, 1.0], "gamma": [0.1], "epsilon": [0.1, 1.0]}

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 5
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 86}

        best_score : 数据类型为'float'
            模型交叉验证得分

    Examples
    --------
        parm_grid = {'n_neighbors': np.arange(n_neighbors_start, n_neighbors_end,n_neighbors_step)}
        best_parm, best_score = findBestParmByGridSearchCv(X, y, estimator=KNeighborsRegressor(), parm_grid=parm_grid)

    Notes
    --------
        knn作为基学习器，base_estimator = KNeighborsRegressor()
        决策树作为基学习器，base_estimator = DecisionTreeRegressor()
        随机森林作为基学习器，base_estimator = RandomForestRegressor()
        svm作为基学习器，base_estimator = SVR()

    """
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(estimator=estimator, param_grid=parm_grid, cv=cv_num, n_jobs=n_jobs).fit(X, y)
    return grid.best_params_, grid.best_score_


def findBestParmByRandomizedSearchCV(X, y, estimator, parm_list, cv_num, n_iter=20, n_jobs=-1):
    """
    随机搜索寻找较优参数

    Parameters
    ----------
        X_train : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y_train : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        estimator :数据类型为'object'
            基学习器对象。
            该对象被认为是实现了sklearn estimator接口的对象。按需求直接使用sklearn中的类对象。
            例如需要使用knn回归来作为基学习器，就必须要导入KNeighborsRegressor这个类，并且将该类的对象赋值给estimator参数,base_estimator=KNeighborsRegressor()。
            具体学习器对象请看下面Notes

        parm_list : 数据类型为'字典'
            待优化参数
            key为需要优化的参数名。value为列表。
            例如parm_grid = {"C": [0.1, 1.0], "gamma": [0.1], "epsilon": [0.1, 1.0]}

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 5
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 86}

        best_score : 数据类型为'float'
            模型交叉验证得分

    Examples
    --------
        parm_grid = {'n_neighbors': np.arange(n_neighbors_start, n_neighbors_end,n_neighbors_step)}
        best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, estimator=KNeighborsRegressor(), parm_grid=parm_grid, cv_num=10, n_iter=20)

    Notes
    --------
        knn作为基学习器，base_estimator = KNeighborsRegressor()
        决策树作为基学习器，base_estimator = DecisionTreeRegressor()
        随机森林作为基学习器，base_estimator = RandomForestRegressor()
        svm作为基学习器，base_estimator = SVR()

    """
    from sklearn.model_selection import RandomizedSearchCV
    rd = RandomizedSearchCV(estimator=estimator, param_distributions=parm_list, n_iter=n_iter, n_jobs=n_jobs,
                            cv=cv_num).fit(X, y)
    return rd.best_params_, rd.best_score_


def svrParm(X, y, C, gama, epsilon, cv_num=3, n_iter=20, n_jobs=-1):
    """
    该方法为SVM回归模型中的三个超参数自动调参。这两个超参数分别是惩罚系数C，RBF核函数的系数γ和损失距离度量ϵ

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        C : 数据类型为'类数组'(list or numpy.narray)
            惩罚系数C，对误差的容忍度。
            可以给定一个确定的量，例如C = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。c越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        gama : 数据类型为'类数组'(list or numpy.narray)
            超参数gama。
            可以给定一个确定的量，例如gama = [0.1]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如gama = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。gama越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        epsilon : 数据类型为'类数组'(list or numpy.narray)
            超参数epsilon。
            可以给定一个确定的量，例如epsilon = [0.1]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如epsilon = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。epsilon越小，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'C': 10.0, 'epsilon': 0.0, 'gamma': 0.7142857142857143}

        best_score : 数据类型为'float'
            模型交叉验证得分

    Examples
    --------
        best_parm, best_score = svrParm(X_train, y_train, C=C, gama=gama, epsilon=epsilon, cv_num=cv_num, n_iter=n_iter)

    """
    from sklearn.svm import SVR
    # {'C': 10.0, 'epsilon': 0.0, 'gamma': 0.7142857142857143}
    parm_list = {"C": C, "gamma": gama, "epsilon": epsilon}
    best_params, best_score = findBestParmByRandomizedSearchCV(X, y, SVR(), cv_num=cv_num, parm_list=parm_list,
                                                               n_iter=n_iter, n_jobs=n_jobs)
    # best_params, best_score = findBestParmByGridSearchCv(X, y, SVR(), parm_list)
    return best_params, best_score


def svcParm(X, y, C=None, gama=None, cv_num=3, n_iter=20, n_jobs=-1):
    """
    该方法为SVM分类模型中的两个超参数自动调参。这两个超参数分别是惩罚系数C和RBF核函数的系数γ

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        C : 数据类型为'类数组'(list or numpy.narray)
            惩罚系数C，对误差的容忍度。
            可以给定一个确定的量，例如C = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。c越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        gama : 数据类型为'类数组'(list or numpy.narray)
            超参数gama。
            可以给定一个确定的量，例如gama = [0.1]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如gama = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]。
            可以从10的幂次序列开始训练，根据效果微调。gama越高，越容易过拟合。过拟合会使得训练样本的准确率高，测试样本准确率低。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 数据类型为'字典'
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'C': 10.0, 'gamma': 0.7142857142857143}

        best_score : 数据类型为'float'
            模型交叉验证得分

    """
    parm_list = {"C": C, "gamma": gama}
    best_params, best_score = findBestParmByRandomizedSearchCV(X, y, SVC(), parm_list=parm_list, cv_num=cv_num,
                                                               n_iter=n_iter, n_jobs=n_jobs)
    # grid = GridSearchCV(SVC(), param_grid={"C": C, "gamma": gama}, cv=cv_num)
    return best_params, best_score


def rfRegressionParm(X, y,
                     n_tree_start, n_tree_end, n_tree_step,
                     max_depth_start, max_depth_end, max_depth_step,
                     min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step,
                     cv_num=3, n_iter=20, n_jobs=-1):
    """
    优化随机森林回归的中决策树的个数

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_tree_start : 数据类型为'int'
            随机森林中决策树的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取较优参数。
            例如n_tree_start = 1

        n_tree_end : 数据类型为'int'
            随机森林中决策树的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取较优参数。
            例如n_tree_start = 200

        n_tree_step : 数据类型为'int'
            每次测试时的步长。默认n_tree_step = 10

        max_depth_start : 数据类型为'int'
            随机森林中树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取较优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            随机森林中树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取较优参数。
            例如max_depth_start = 50

        max_depth_step : 数据类型为'int'
            每次测试时的步长。默认max_depth_step = 1
            例如当步长max_depth_step = 3时，左端点max_depth_start = 1，右端点max_depth_end = 10，此时树的最大深度就在[1, 4, 7]中取一个较优值。

        min_samples_leaf_start : 数据类型为'int'
            叶子节点最少的样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取较优参数。
            例如min_samples_leaf_start = 1。

        min_samples_leaf_end : 数据类型为'int'
            叶子节点最少的样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取较优参数。
            例如min_samples_leaf_start = 100

        min_samples_leaf_step : 数据类型为'int'
            每次测试时的步长。默认min_samples_leaf_step = 1
            例如当步长min_samples_leaf_step = 3时，左端点min_samples_leaf_step = 1，右端点min_samples_leaf_step = 10，此时叶子节点最少的样本数就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 191, 'min_samples_leaf': 3, 'max_features': 'log2', 'max_depth': 16}

        best_score : float
            模型交叉验证得分

    Examples
    --------
        best_parm, best_score = rfRegressionParm(X_train, y_train,
                                                1, 200, 2,
                                                1, 50, 1,
                                                1, 20, 1,
                                                10, 20)

    """
    import numpy as np
    from sklearn import ensemble
    parm_grid = {'n_estimators': np.arange(n_tree_start, n_tree_end, n_tree_step),
                 'max_depth': np.arange(max_depth_start, max_depth_end, max_depth_step),
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'min_samples_leaf': np.arange(min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step)
                 }
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, estimator=ensemble.RandomForestRegressor(),
                                                             parm_list=parm_grid, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    return best_parm, best_score


def dtRegressionParm(X, y, max_depth_start, max_depth_end, max_depth_step, cv_num=3, n_iter=20, n_jobs=-1):
    """
    优化CART回归中树的最大深度

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        max_depth_start : 数据类型为'int'
            树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如max_depth_end = 50

        max_depth_step : 数据类型为'int'
            每次测试时的步长。默认max_depth_step = 1
            例如当步长max_depth_step = 3时，左端点max_depth_start = 1，右端点max_depth_end = 10，此时树的最大深度就在[1, 4, 7]中取一个最优值

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : 数据类型为'int'
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'max_depth': 16}

        best_score : float
            模型交叉验证得分

    """
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    parm_list = {'max_depth': np.arange(max_depth_start, max_depth_end, max_depth_step)}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, estimator=DecisionTreeRegressor(),
                                                             parm_list=parm_list, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    return best_parm, best_score


def knnRegressionParm(X, y, n_neighbors_start, n_neighbors_end, n_neighbors_step, cv_num, n_jobs):
    """
    优化knn回归中的K值

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_neighbors_start : 数据类型为'int'
            K值。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_neighbors_start = 1

        n_neighbors_end : 数据类型为'int'
            K值。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_neighbors_end = 50

        n_neighbors_step : 数据类型为'int'
            每次测试时的步长。默认n_neighbors_step = 1
            例如当步长n_neighbors_step = 3时，左端点n_neighbors_start = 1，右端点n_neighbors_end = 10，此时K值就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_jobs : 数据类型为'int'
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_neighbors': 5}

        best_score : float
            模型交叉验证得分

    """
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np
    parm_grid = {'n_neighbors': np.arange(n_neighbors_start, n_neighbors_end, n_neighbors_step)}
    best_parm, best_score = findBestParmByGridSearchCv(X, y, estimator=KNeighborsRegressor(), parm_grid=parm_grid,
                                                       cv_num=cv_num, n_jobs=n_jobs)
    return best_parm, best_score


def baggingRegressionParm(X, y, base_estimator, n_estimators_start, n_estimators_end, n_estimators_step, cv_num=3,
                          n_jobs=-1):
    """
    优化bagging的中基学习器的数量

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        base_estimator : 数据类型为'object'
            基学习器对象
            该对象被认为是实现了sklearn estimator接口的对象。按需求直接使用sklearn中的类对象。
            例如需要使用knn回归来作为基学习器，就必须要导入KNeighborsRegressor这个类，并且将该类的对象赋值给estimator参数,base_estimator=KNeighborsRegressor()。
            具体学习器对象请看下面Notes

        n_estimators_start : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_estimators_start = 1


        n_estimators_end : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_estimators_end = 200

        n_estimators_step : 数据类型为'int'
            每次测试时的步长。默认n_estimators_step = 5
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 140}

        best_score : float
            模型交叉验证得分

    Notes
    --------
        knn作为基学习器，base_estimator = KNeighborsRegressor()
        决策树作为基学习器，base_estimator = DecisionTreeRegressor()
        随机森林作为基学习器，base_estimator = RandomForestRegressor()
        svm作为基学习器，base_estimator = SVR()

    """
    from sklearn import ensemble
    parm_grid = {'n_estimators': np.arange(n_estimators_start, n_estimators_end, n_estimators_step)}
    best_parm, best_score = findBestParmByGridSearchCv(X, y, estimator=ensemble.BaggingRegressor(
        base_estimator=base_estimator), parm_grid=parm_grid, cv_num=cv_num, n_jobs=n_jobs)
    return best_parm, best_score


def adaBoostRegressionParm(X, y, n_estimators_start, n_estimators_end, n_estimators_step, max_depth_start,
                           max_depth_end, max_depth_step, cv_num=3, n_iter=20, n_jobs=-1):
    """
    优化adaBoost中的基学习器CART树、基学习器的数量、学习率和loss函数

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_estimators_start : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_estimators_start = 1

        n_estimators_end : 数据类型为'int'
            基学习器的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_estimators_end = 200

        max_depth_start : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如max_depth_end = 50

        max_depth_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        n_estimators_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : int
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'n_estimators': 152, 'learning_rate': 0.1, 'loss': 'linear'}

        best_score : float
            模型交叉验证得分

    """
    from sklearn import ensemble
    from sklearn.tree import DecisionTreeRegressor
    dt_parm, dt_score = dtRegressionParm(X, y, max_depth_start, max_depth_end, max_depth_step)
    dt_model = DecisionTreeRegressor(max_depth=dt_parm['max_depth']).fit(X, y)
    parm_list = {'n_estimators': np.arange(n_estimators_start, n_estimators_end, n_estimators_step),
                 'learning_rate': [0.01, 0.05, 0.1, 0.5, 1, 1.5],
                 'loss': ['linear', 'square', 'exponential']}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, ensemble.AdaBoostRegressor(base_estimator=dt_model),
                                                             parm_list=parm_list, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    best_parm['model'] = dt_model
    return best_parm, best_score


def gradientBoostingParm(X, y,
                         n_estimators_start, n_estimators_end,
                         max_depth_start, max_depth_end,
                         min_samples_leaf_start, min_samples_leaf_end,
                         min_samples_split_start, min_samples_split_end,
                         learning_rate, subsample,
                         max_depth_step, n_estimators_step, min_samples_leaf_step, min_samples_split_step,
                         cv_num=3, n_iter=20, n_jobs=-1):
    """
    优化GBRT中基学习器的数量和CART树的深度

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        n_estimators_start : 数据类型为'int'
            基学习器(CART)的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如n_estimators_start = 1

        n_estimators_end : 数据类型为'int'
            基学习器(CART)的数量。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如n_estimators_end = 200

        max_depth_start : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            例如max_depth_start = 1

        max_depth_end : 数据类型为'int'
            CART树的最大深度。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的右端点。程序将会在区间内选取最优参数。
            例如max_depth_end = 50

        min_samples_leaf_start :  数据类型为'int'
            叶子节点最小样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_leaf一开始设置为1。如果样本量数量非常大，则推荐增大这个值。
            例如max_depth_start = 1

        min_samples_leaf_end : 数据类型为'int'
            叶子节点最小样本数。
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_leaf一开始设置为1。如果样本量数量非常大，则推荐增大这个值。
            例如max_depth_end = 2

        min_samples_split_start : 数据类型为'int'
            内部节点再划分所需最小样本数
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_split一开始设置为2。如果样本量数量或者特征数非常大，则推荐增大这个值。
            例如max_depth_start = 2

        min_samples_split_end : 数据类型为'int'
            内部节点再划分所需最小样本数
            给定一个可能取值的区间(左闭右开)，该参数为这个区间的左端点。程序将会在区间内选取最优参数。
            建议min_samples_split一开始设置为2。如果样本量数量或者特征数非常大，则推荐增大这个值。
            例如max_depth_end = 3

        learning_rate : 数据类型为'类数组'(list or numpy.narray)
            每个弱学习器的学习步长。
            可以给定一个确定的量，例如learning_rate = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如learning_rate = [0.001, 0.01, 0.1, 1.0]。
            可以从10的幂次序列开始训练，根据效果微调。较小的ν意味着我们需要更多的弱学习器的迭代次数。

        subsample : 数据类型为'类数组'(list or numpy.narray)
            子采样。
            可以给定一个确定的量，例如subsample = [1.0]。
            如果不确定的话，也可以给定一个'类数组'，其中包含多个可能的取值，程序会自动选取最优值。例如subsample = [0.5, 0.6, 0.7, 0.8]。
            选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐[0.5, 0.8]之间

        max_depth_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        n_estimators_step : 数据类型为'int'
            每次测试时的步长。
            例如当步长n_estimators_step = 3时，左端点 n_estimators_start = 1，右端点n_estimators_end = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        min_samples_leaf_step : 数据类型为'int'
            每次测试时候的步长。默认min_samples_leaf_step = 1
            例如当步长min_samples_leaf_step = 3时，左端点min_samples_leaf_step = 1，右端点min_samples_leaf_step = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        min_samples_split_step : 数据类型为'int'
            每次测试时候的步长。默认min_samples_split_step = 1
            例如当步长min_samples_split_step = 3时，左端点min_samples_split_step = 1，右端点min_samples_split_step = 10，此时基学习器的数量就在[1, 4, 7]中取一个较优值。

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : int
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'subsample': 0.5, 'n_estimators': 116, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 29, 'loss': 'quantile', 'learning_rate': 0.1}

        best_score : float
            模型交叉验证得分

    """
    from sklearn import ensemble
    parm_list = {'n_estimators': np.arange(n_estimators_start, n_estimators_end, n_estimators_step),
                 'max_depth': np.arange(max_depth_start, max_depth_end, max_depth_step),
                 'loss': ['ls', 'lad', 'huber', 'quantile'],
                 'learning_rate': learning_rate,
                 'subsample': subsample,
                 'min_samples_leaf': np.arange(min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step),
                 'min_samples_split': np.arange(min_samples_split_start, min_samples_split_end, min_samples_split_step),
                 'max_features': ['log2', 'sqrt', None]}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, ensemble.GradientBoostingRegressor(),
                                                             parm_list=parm_list, cv_num=cv_num, n_iter=n_iter,
                                                             n_jobs=n_jobs)
    return best_parm, best_score


def lassoParm(X, y, alpha, max_iter, cv_num=3, n_iter=20, n_jobs=-1):
    """
    lasso回归参数调优

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            训练数据。
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            目标值(真实值)。
            每一个参数是对应样本数据的目标值
            例如y_train = [16.50000,31.10000,10.50000]

        alpha : float 或者 '类数组'
            正则项系数。数值越大，则对复杂模型的惩罚力度越大。'类数组'是拥有数组结构，可以转化为narray的任何python对象。
            传入数组序列时，程序会在其中自动选优

        max_iter : int
            最大迭代次数

        cv_num : 数据类型为'int'
            S折交叉验证的折数。默认cv_num = 3
            即将训练集分成多少份来进行交叉验证。如果样本较多的话，可以适度增大cv的值

        n_iter : int
            训练次数，次数越大精度越高。默认n_iter=20

        n_jobs : int
            用来设定cpu的运行情况。默认n_jobs = -1为使用全部cpu

    Returns
    -------
        best_parm : 字典
            模型最优参数。key为参数名称，value为对应值。
            例如best_parm = {'alpha': 0.1}

        best_score : float
            模型交叉验证得分

    """
    from sklearn.linear_model import Lasso
    parm_list = {'alpha': alpha}
    best_parm, best_score = findBestParmByRandomizedSearchCV(X, y, Lasso(max_iter=max_iter), parm_list=parm_list,
                                                             cv_num=cv_num, n_iter=n_iter, n_jobs=n_jobs)
    return best_parm, best_score


def pearsonrValidation(X, y, X_name=None):
    """
    皮尔逊相关系数

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            自变量
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            因变量
            例如y_train = [16.50000,31.10000,10.50000]

        X_name : 形状为(特征数量, )的'类数组'。可选
            特征名。默认X_name = None

    Returns
    -------
        retVal : DataFrame
            若给没有给定参数X_name时候，返回一个(特征数量, 2)的DataFrame，第一列为皮尔逊系数，第二列为p值。
            若给定参数X_name时候，返回一个(特征数量, 3)的DataFrame，首列为特征名，第二列为皮尔逊系数，第三列为p值。

    """
    from pandas import DataFrame
    from scipy.stats import pearsonr
    import numpy as np
    res = []
    for i in range(X.shape[1]):
        res.insert(i, pearsonr(X[:, i], y))
    res = np.array(res)
    transVal = {'pearsonrCoef': res[:, 0],
                'pVal': res[:, 1]}
    retVal = DataFrame(transVal)
    if X_name is not None:
        # retVal.loc[:, 'd'] = X_name
        retVal.insert(loc=0, column='featureName', value=X_name)
    return retVal


def rocDisplay(y_true, y_pred):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, RocCurveDisplay
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    dis = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    dis.plot()
    plt.show()


def confusionMatrixDisplay(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    confusionMatrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
    ax = plt.figure().subplots()
    ax.set(title="ours")
    disp.plot(ax=ax)
    plt.show()


def FeatureCorrelation(X, y, X_name=None, img=None):
    """
    相关性分析

    Parameters
    ----------
        X : 数据类型为'类数组'(list or numpy.narray)或'矩阵'(numpy.matrix)
            自变量
            数组和矩阵的每一行都是一条样本数据，每一列代表不同的特征。
            例如X_train = [[5.63100,6.80000,5.81800],[6.80000,5.81800,5.42700],[5.81800,5.42700,6.12900]]

        y : 数据类型为'类数组'(list or numpy.narray)
            因变量
            例如y_train = [16.50000,31.10000,10.50000]

        X_name : 形状为(特征数量, )的'类数组'。可选
            特征名。默认X_name = None

        img : boolean
            如果需要将自变量和因变量的线性关系可视化，请让img=True，否则请保持默认img=None。

    """
    import matplotlib.pyplot as plt
    ps = pearsonrValidation(X, y, X_name)
    print(ps)
    if img is not None:
        for i in range(X.shape[1]):
            plt.scatter(X[:, i], y)
            if X_name is None:
                plt.show()
            else:
                plt.title(X_name[i])
                plt.show()


def arimaModel(data, d, pmax, qmax):
    """
    ARIMA模型。模型定阶选用BIC检验，组合各种p和q，获取最小BIC值的p和q

    Parameters
    ----------
        data : 数据类型为'numpy.narray'
            原始数据序列。为一维数组。为非平稳序列
            每一个参数为该时间段的值。
            例如# data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422]

        d : 数据类型为'int'
            差分阶数。
            将序列变成平稳性序列。推荐使用该函数前先对原始数据进行差分，观测数据平稳性。

        pmax : 数据类型为'int'
            AR(p)的最大值。一般不超过（data长度 / 10）

        qmax : 数据类型为'int'
            MA(q)的最大值。一般不超过（data长度 / 10）

    Returns
    -------
        model : object
            训练好的ARIMA模型。可以通过model.forecast(step)来进行预测，返回的结果包括：预测结果、标准误差、置信区间

    Examples
    -------
        data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
                6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
                10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
                12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
                13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
                9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
                11999,9390,13481,14795,15845,15271,14686,11054,10395]
        data = np.array(data, dtype=np.float)
        arimamodel = arimaModel(data, 1, np.int(data.__len__() / 10), np.int(data.__len__() / 10))
        # 想要预测未来五个时间单位的值
        arimamodel.forecast(5)

    """
    from statsmodels.tsa.arima_model import ARIMA
    data = np.array(pd.DataFrame(data).dropna()).flatten()
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(data, (p, d, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().astype(float).idxmin()
    model = ARIMA(data, (p, d, q)).fit()
    # print(model.forecast(5))
    return model


def armaModel(data, pmax, qmax):
    """
    ARMA模型。模型定阶选用BIC检验，组合各种p和q，获取最小BIC值的p和q

    Parameters
    ----------
        data : 数据类型为'numpy.narray'
            原始数据序列。为一维数组。要求为平稳序列
            每一个参数为该时间段的值。
            例如# data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422]

        pmax : 数据类型为'int'
            AR(p)的最大值。一般不超过（data长度 / 10）

        qmax : 数据类型为'int'
            MA(q)的最大值。一般不超过（data长度 / 10）

    Returns
    -------
        model : object
            训练好的ARMA模型。可以通过model.forecast(step)来进行预测，返回的结果包括：预测结果、标准误差、置信区间

    Examples
    -------
        data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
                6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
                10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
                12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
                13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
                9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
                11999,9390,13481,14795,15845,15271,14686,11054,10395]
        data = np.array(data, dtype=np.float)
        armamodel = armaModel(np.diff(data, 1), np.int(data.__len__() / 10), np.int(data.__len__() / 10))
        # 想要预测未来五个时间单位的值
        arimamodel.forecast(5)

    """
    from statsmodels.tsa.arima_model import ARMA
    data = np.array(pd.DataFrame(data).dropna()).flatten()
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARMA(data, (p, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().astype(float).idxmin()
    model = ARMA(data, (p, q)).fit()
    # print(model.forecast(5))
    return model


class GraySystem:
    """
    该类主要包括灰色关联度分析和多因素灰色预测模型

    """

    def AGO(self, m):
        """
        累加生成算子

        Parameters
        ----------
            m : 数据类型为'numpy.narray'
                为一维数组
                需要进行累加操作的序列。
                例如m = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        Returns
        -------
            li : 数据类型为'numpy.narray'
                为一维数组
                累加完成后的序列
                x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        """
        import numpy as np
        li = np.zeros(shape=m.shape)
        for i in range(m.__len__()):
            temp = 0
            for j in range(i + 1):
                temp = m[j] + temp
            li[i] = temp
        return li

    def MEAN(self, m):
        """
        均值生成算子

        Parameters
        ----------
            m : 数据类型为'numpy.narray'
                为一维数组
                需要生成邻均值序列的序列。
                例如m = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        Returns
        -------
            li : 数据类型为'numpy.narray'
                为一维数组
                邻均值序列
                x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]

        """
        import numpy as np
        li = np.zeros(shape=m.shape)
        for i in range(m.__len__()):
            if i == 0:
                continue
            li[i] = 0.5 * (m[i] + m[i - 1])
        li = np.delete(li, 0)
        return li

    def greyRelationAnalysis(self, X, y):
        """
        灰色关联度分析

        Parameters
        ----------
        X : 数据类型为'numpy.narray'
            比较队列。列向量表示X = [x0, x1, x2,]
            数组每一列都是一条相关因素序列，每一行代表某个时间节点该因素的值。
            列向量表示X = [x0, x1, x2, x3]
            即X = [[104.00000,135.60000,131.60000,54.20000],
                    [101.80000,140.20000,135.50000,54.90000],
                    [105.80000,140.10000,142.60000,54.80000],
                    [111.50000,146.90000,143.20000,56.30000],
                    [115.97000,144.00000,142.20000,54.50000]]

        y : 数据类型为'numpy.narray'
            参考队列。列向量
            每一个值都是目标在该时间节点，多个因素影响下的结果
            例如y = [[560823],
                    [542386],
                    [604834],
                    [591248],
                    [583031]]

        Returns
        -------
            r : 形状为(特征数量, )的'numpy类型数组'
                关联度数组
                例如r = [0.72135605 0.69545271 0.65791122 0.73256354]

        Examples
        -------
            a = [560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677]
            x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]
            x1 = [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8]
            x2 = [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5]
            x3 = [54.2,54.9,54.8,56.3,54.5,54.6,54.9,54.8,49.3,41.5,48.9]
            r = gs.greyRelationAnalysis(np.array(mat([x0, x1, x2, x3]).T), np.array(mat(a).T))
            # print(r)结果为r = [0.72135605 0.69545271 0.65791122 0.73256354]

        """
        import numpy as np
        y = np.array(y, dtype=float).ravel()
        X_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / X_mean[i]
        for i in range(y.shape[0]):
            y[i] = y[i] / y_mean
        new = []
        for i in range(X.shape[1]):
            new.append(np.array(np.mat(X[:, i] - y)))
        new = np.array(np.mat(np.array(new)).T)
        # new = np.array(mat(new).T)
        mmax = np.max(np.max(new.__abs__(), axis=0))
        mmin = np.min(np.min(new.__abs__(), axis=0))
        rho = 0.5
        # 关联系数
        ksi = ((mmin + rho * mmax) / (new.__abs__() + rho * mmax))
        # 关联度
        r = sum(ksi, axis=0) / ksi.shape[0]
        return r

    def greyPredictiveParm(self, X_original, y_original):
        """
        获得发展系数和驱动系数

        Parameters
        ----------
            X_original : 形状为(相关因素序列数量， 每个因素包含的数量)的numpy类型数组
                相关因素。
                数组的每一行代表一条相关因素序列，每一列代表某个时间节点该因素的值。

            y_original : 形状为(系统特征数据数量, )的numpy类型数组
                系统特征数据序列(真实值)。
                每一个值都是目标在该时间节点，多个因素影响下的结果

        Returns
        -------
            al :  float
                GM(1, n)模型中的发展系数

            b : 形状为(相关因素个数， )的numpy类型数组
                GM(1, n)模型中的驱动系数列表。
                每一个值代表对应相关因素序列的驱动系数。

        """
        import numpy as np
        xi = []
        for i in range(X_original.shape[0]):
            xi.append(self.AGO(X_original[i, :]))
        xi = np.array(xi)
        z = self.MEAN(np.array(self.AGO(y_original)))
        Y = y_original.reshape(-1, 1)
        Y = np.delete(Y, 0)
        Y = np.mat(Y)
        B = np.array(np.mat(xi[:, 1:]).T)
        B = np.insert(B, 0, -z, 1)
        B = np.mat(B)
        theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y.T)
        al = theat[0, 0]
        b = theat[1:, :].T
        b = np.array(b).flatten()
        return al, b

    def greyPredictiveModel(self, X_new, al, b, y_first):
        """
        多因素灰色预测模型

        Parameters
        ----------
            X_new : 数据类型为'numpy.narray'
                新的相关因素。列向量表示X = [x0, x1, x2,]
                数组每一列都是一条相关因素序列，每一行代表某个时间节点该因素的值。
                列向量表示X = [x0, x1, x2, x3]
                即X = [[104.00000,135.60000,131.60000,54.20000],
                        [101.80000,140.20000,135.50000,54.90000],
                        [105.80000,140.10000,142.60000,54.80000],
                        [111.50000,146.90000,143.20000,56.30000],
                        [115.97000,144.00000,142.20000,54.50000]]

            al : float
                GM(1, n)模型中的发展系数

            b : 形状为(相关因素个数， )的numpy类型数组
                GM(1, n)模型中的驱动系数列表。
                每一个值代表对应相关因素序列的驱动系数。

            y_first : float
                预测值的起点

        Returns
        -------
            G : 形状为(系统特征数据数量, )的numpy类型数组
                系统特征数据序列(预测值)。
                每一个值都是目标在该时间节点，多个因素影响下的结果。

        Examples
        -------
            # 根据已知数据得出参数al, b
            a = [560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677]
            x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]
            x1 = [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8]
            x2 = [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5]
            x3 = [54.2,54.9,54.8,56.3,54.5,54.6,54.9,54.8,49.3,41.5,48.9]
            al, b = gs.greyPredictiveParm(np.array([x0, x1, x2, x3]), np.array(a))

            # 当有新的相关因素产生时可以预测
            x_0 = [113.57, 112.03, 128.3, 106.4, 113.1, 71.4, 92.5]
            x_1 = [139.2, 135.2, 145.1, 141.5, 142, 141.6, 136.3]
            x_2 = [138, 131.5, 142.6, 138.4, 142.2, 131, 134.3]
            x_3 = [56.1,55.9,51.8,58.3,53.4,56.7,56.2]
            G = gs.greyPredictiveModel(np.array([x_0, x_1, x_2, x_3]), al, b, a[len(a) - 1])

        """
        import numpy as np
        xi = []
        for i in range(X_new.shape[0]):
            xi.append(self.AGO(X_new[i, :]))
        xi = np.array(xi)
        U = []
        for j in range(xi.shape[1]):
            sum1 = 0
            for i in range(xi.shape[0]):
                sum1 += np.mat(xi)[i, j] * b[i]
            U.append(sum1)
        F = []
        f = 1
        F.append(y_first)
        while f < X_new.shape[1]:
            F.append((y_first - U[f] / al) * np.exp(-al * f) + U[f] / al)
            f = f + 1
        F = np.array(F).flatten()
        G = [y_first]
        g = 1
        while g < X_new.shape[1]:
            G.append(F[g] - F[g - 1])
            g += 1
        G = np.array(G)
        return G


def find_max_region(file, mbKSize=15, denoiseScheme='other', topMargin=15, bottomMargin=20):
    """
    最大连通区域
    Parameters
    ----------
    file : narray
        图片数据
    mbKSize : int
        中值滤波核大小.
        作用为平滑图像
    denoiseScheme : string
        当镜面反射和下表面有接触的时候，连通区域可能会包含镜面反射内容，这个时候选用'mediaBlur'模式
        没有上述情况就保持'default'模式
    topMargin
    bottomMargin

    Returns
    -------

    """
    import cv2
    import numpy as np

    afterDenoise = None
    if denoiseScheme == 'default':
        # 不需要对镜面反射做处理
        afterDenoise = denoise(img=file, n=15, mb_size=mbKSize, q=35)
    elif denoiseScheme == 'mediaBlur':
        # 中值滤波下需要对镜面反射做处理
        colNum = file.shape[1]
        for col in range(colNum - 1):
            for i in range(topMargin):
                file[i][col] = 0
            for j in range(colNum - bottomMargin - 1, colNum):
                file[j][col] = 0
        afterDenoise = cv2.medianBlur(file, mbKSize)
    elif denoiseScheme == 'other':
        afterDenoise = cv2.medianBlur(file, mbKSize)
    temp = np.copy(afterDenoise)
    ret, threshold = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # stats对应的是x,y,width,height和面积， centroids为中心点， labels表示各个连通区在图上的表示
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    # 选出背景的序号
    labelNum = 0
    for i in range(stats.shape[0]):
        if stats[i, 0] == 0 and stats[i, 1] == 0:
            labelNum = i
            break
    stats = np.delete(stats, [0], axis=0)
    num_labels = num_labels - 1
    # 将label列表中labelNum组全部置为0背景
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == labelNum:
                labels[i][j] = 0
    output = np.zeros((threshold.shape[0], threshold.shape[1]), np.uint8)
    # 将图中的连通区域组合起来
    for i in range(1, num_labels + 1):
        mask = labels == i
        output[:, :][mask] = 255
    return output


def binaryMask(mask_file):
    """
    掩膜二值化
    :param mask_file:
    :return:
    """
    for i in range(mask_file.shape[0]):
        for j in range(mask_file.shape[1]):
            mask_file[i][j] = 1 if (mask_file[i][j] == 255) else 0


def bow(descriptor_list, k, label):
    """
    词袋算法
    :param descriptor_list: 图片数据集的特征描述子列表
    :param k: 要将特征描述子分成几类
    :param label: 图片标签
    :return:
    """
    from sklearn.cluster import KMeans
    import numpy as np
    ret = []
    for item in descriptor_list:
        item = item.reshape(-1, 1)
        # 先测出一个item的最佳k值
        y_pre = KMeans(n_clusters=k, random_state=32).fit_predict(item)
        l = len(y_pre)
        unique, counts = np.unique(y_pre, return_counts=True)
        frequency = dict(zip(unique, counts / l))
        ret.append(list(frequency.values()))
    ret = np.insert(np.array(np.mat(ret)), k, label, axis=1)
    return ret


def surfaceFitting(img, deg: int = 2, mbSize: int = 15, model: object = None, denoiseScheme: str = 'default',
                   manualLoc: list = None):
    """
    通过连通区域来进行表面拟合
    Parameters
    ----------
    denoiseScheme
    img : narray
        图像数据
    deg : int
        拟合曲线的阶数,只有在model=None的时候才生效
    mbSize : int
        中值滤波核的大小，用来平滑图像
    model : object
        预训练的回归模型，默认为None
    manualLoc

    Returns
    -------
    surfacePosition : array_like
        确定表面位置
    location : array_like
        预估计表面位置
    Notes
    -----
    根据图像的特性来调参，若在种子OCT图像中发现有断层出现在内部，这个时候连通区域可能会出现较大的缺损，
    可以使用一阶函数去拟合。

    """
    import numpy as np
    import cv2
    location = []
    if manualLoc is None:
        # 找出最大连通区域
        region = find_max_region(img, mbSize, denoiseScheme=denoiseScheme)
        region = cv2.erode(region, kernel=(3, 3), iterations=5)
        # 从下往上找出每一列中第一个大于0的值， 存入到location数组中
        col = region.shape[1]
        row = region.shape[0]
        for i in range(col):
            temp = region[:, i]
            j = row - 1
            while j > 0:
                if temp[j] > 0:
                    location.append((j, i + 1))
                    break
                j -= 1
    else:
        location = manualLoc
    # 将location数组中的值当做目标值，将x横坐标当做自变量，进行拟
    location = np.array(location)
    X = location[:, 1]
    y = location[:, 0]
    X_test = [i + 1 for i in range(img.shape[1])]
    if model is not None:
        surfacePosition = model.predict(X_test)
    else:
        z1 = np.polyfit(X, y, deg)
        p1 = np.poly1d(z1)
        surfacePosition = np.array(p1(X_test), dtype=np.int32)
    for idx, item in enumerate(surfacePosition):
        if item > img.shape[0]:
            surfacePosition[idx] = img.shape[0] - 1
    return surfacePosition, location


def flatten(img, surfacePosition):
    """
    展平图片
    Parameters
    ----------
    img : 数据类型为numpy.narray
        图片数据

    surfacePosition ：numpy.array
        表面位置索引数组，由列向量索引组成，长度等于图片的列向量数量
        例如np.array([224, 212, 264, 203 ....，123])

    Returns
    -------
    ret：数据类型为numpy.narray
        展平后的图片
    """
    import numpy as np
    ret = np.zeros(shape=img.shape, dtype=np.uint8)
    imax = np.max(surfacePosition)
    imin = np.min(surfacePosition)
    mid = int((imax + imin) / 2)
    for idx, value in enumerate(surfacePosition):
        diff = np.abs(value - mid)
        temp = img[:, idx]
        if value > mid:
            k = diff
            for j in range(temp.size - diff):
                ret[k][idx] = temp[j]
                k += 1
        else:
            k = 0
            for j in range(diff, temp.size):
                ret[k][idx] = temp[j]
                k += 1
    return ret


def denoise(img, n: int, mb_size: int = 3, q: int = 30, is_top: bool = True):
    """
    图片去噪
    Parameters
    ----------
    is_top
    q
    mb_size : int
        中值滤波核大小，作用为平滑图像边缘，一般选取小滤波核
    img:数据类型为numpy.narray
        图片数据

    n:数据类型为int
        使用前n行的均值和标准差进行阈值去噪

    Returns
    -------
        ret:数据类型为numpy.narray
            返回图片

    """
    import numpy as np
    import cv2
    ret = np.copy(img)
    k = []
    num = 0
    sum = 0
    if is_top:
        a = 0
        b = n
    else:
        a = ret.shape[0] - n
        b = ret.shape[1]
    for i in range(a, b):
        for j in range(ret.shape[1]):
            if ret[i][j] <= 3:
                continue
            sum += ret[i][j]
            k.append(ret[i][j])
            num += 1
    ave = np.divide(sum, num)
    # ave = sum / num
    sum = 0
    for value in k:
        sum += np.power(value - ave, 2)
    sd = np.sqrt(sum / num)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i][j] <= sd + ave:
                ret[i][j] = 0
    for i in range(q):
        for j in range(ret.shape[1]):
            ret[i][j] = 0
            ret[ret.shape[0] - 1 - i][j] = 0
    if mb_size > 0:
        ret = cv2.medianBlur(ret, ksize=mb_size)
    return ret


def cropImg(img, top=0, button=0, left=0, right=0):
    """
    裁剪图片
    Parameters
    ----------
        img:数据类型为numpy.narray
            图片数据

        top:数据类型为int
            裁剪掉距离顶部的k行
            默认值为0

        button:数据类型为int
            裁剪掉距离底部的k行
            默认值为0

        left:数据类型为int
            裁剪掉距离左边的k列
            默认值为0

        right:数据类型为int
            裁剪掉距离右边的k列
            默认值为0

    Returns
    -------
        ret:数据类型为numpy.narray
            裁剪完的图片

    Examples
    --------
        img = cv2.imread('7-462.jpg', flags=0)
        ret = cropImg(img, 60)
        cv2.imshow('test', ret)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    ret = img[top:img.shape[0] - button, left:img.shape[1] - right]
    return ret


def copyD2D(srcDir: str, dstDir: str):
    """
    將一個文件夾中的所有圖片copy到制定目錄
    Parameters
    ----------
        srcDir: String
            源目錄

        dstDir: String
            目標目錄

    """
    import os
    from shutil import copy
    srcNames = os.listdir(srcDir)
    for item in srcNames:
        copy(srcDir + item, dstDir)
        print(dstDir + item + ' done!')


def get_data(feature_path: str):
    """
    从特征向量文件（带标签）中分离出特征和标签
    Parameters
    ----------
    feature_path ：String
        特征向量的路径

    Returns
    -------
        X：numpy.narray
            特征向量
        y：float
            真实值

    """
    import numpy as np
    with open(feature_path, 'r') as f:
        contents = f.readlines()
        for i in range(len(contents)):
            contents[i] = contents[i].strip().split(',')
        contents = np.array(contents)
    X = contents[:, :-1].astype(np.float64)
    y = contents[:, -1].astype(np.float64)
    return X, y


def standardization(img):
    """
    对图像进行标准化
    Parameters
    ----------
    img

    Returns
    -------

    """
    import numpy as np
    import cv2
    row = img.shape[0]
    col = img.shape[1]
    ret = np.copy(img)
    temp = cv2.medianBlur(img, ksize=15)
    max_temp = np.max(temp)
    max = np.max(ret[39:-25, :])
    min = np.min(ret[39:-25, :])
    Im = 1.05 * max_temp
    for i in range(row):
        for j in range(col):
            if ret[i][j] <= Im:
                ret[i][j] = Im * (ret[i][j] - min) / (max - min)
            else:
                ret[i][j] = Im
    return ret


def startFlatten(root_path: str, dir_list: list, log_path: str, save_path: str, crop: list, deg: int = 3,
                 mbKSize: int = 11,
                 denoiseScheme: str = 'default'):
    """
    对一个根目录下的所有类别OCT图像，进行一种展平操作
    Parameters
    ----------
    crop
    root_path
    dir_list
    log_path
    save_path
    deg
    mbKSize
    denoiseScheme

    Returns
    -------
    Notes
    -----
    只对一部分类别图像进行操作,下面例子中dataset中有class1类别图像，在其中再创建两个文件夹done和failed，将一部分图片复制到failed中，并且将
    img_root_path = root_path + dir改成img_root_path = root_path + dir + '/failed/'。然后test.py中调用，
    startFlatten(root_path='../sources/dataset/', dir_list=['12'], log_path='log.txt',
                 save_path='../sources/dataset/12/done/', deg=1, mbKSize=9)
    目录结构:
    project:
        sources:
            dataset:
                class1:
                    done:
                    failed:
                    1.jpg
                    ...
                ...
        src:
            test.py
    如果需要看 原图 去噪图 掩膜 预估计点 拟合点 展平后的图像排列
    ret = ut.standardization(img)
    region, afterDenoise = ut.find_max_region(ret, 9, denoiseScheme='default')
    y, _ = ut.surfaceFitting(ret, deg=2, mbSize=9)
    fitted_location_img = np.copy(img)
    original_location_img = np.copy(img)
    for i in range(fitted_location_img.shape[1]):
        fitted_location_img[y[i]][i] = 255
    for i in range(len(_)):
        k = _[i]
        original_location_img[k[0]][k[1] - 1] = 255
    flattened_img = ut.flatten(img, [512 - i for i in y])
    merge = np.hstack((img, ret, region, original_location_img, fitted_location_img, flattened_img))
    cv2.imwrite('14_181.jpg', merge)

    """
    import time
    import os
    import cv2
    with open(log_path, 'a+') as f:
        for dir in dir_list:
            img_root_path = root_path + dir
            img_list = os.listdir(img_root_path)
            print(dir + '开始：')
            f.write(dir + '开始：')
            f.write('\n')
            for idx, img_name in enumerate(img_list):
                try:
                    if img_name[-3:] != 'jpg':
                        continue
                    img_path = img_root_path + img_name
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    img = cv2.imread(img_path, flags=0)
                    time_start = time.time()
                    ret = standardization(img)
                    y, _ = surfaceFitting(ret, deg=deg, mbSize=mbKSize, denoiseScheme=denoiseScheme)
                    img = flatten(img, [512 - i for i in y])
                    img = cropImg(img, crop[0], crop[1], crop[2], crop[3])
                    cv2.imwrite(save_path + '1-' + img_name, img)
                    end_time = time.time()
                    processing_time = abs(time_start - end_time)
                    print('total: {} current: {} '.format(len(img_list),
                                                          idx + 1) + img_name + ' done, processing time is: ' + processing_time.__str__() + ' s')
                    f.write('total: {} current: {} '.format(len(img_list),
                                                            idx + 1) + img_name + ' done, processing time is: ' + processing_time.__str__() + ' s')
                    f.write('\n')
                except Exception as e:
                    print('total: {} current: {} '.format(len(img_list),
                                                          idx + 1) + img_name + ' failed, processing time is: ' + processing_time.__str__() + ' s')
                    f.write('total: {} current: {} '.format(len(img_list),
                                                            idx + 1) + img_name + ' failed' + 'reason: ' + e.__str__() + ', processing time is: ' + processing_time.__str__() + ' s')
                    f.write('\n')
            print(dir + '结束\n')
            f.write(dir + '结束\n')
            f.write('\n')


class TestModel:
    """
    模型測試的類
    """

    def __init__(self, model, modelLocation, strict=False):
        model.load_state_dict(torch.load(modelLocation), strict=strict)
        self.model = model

    def testSingleImg(self, img, visualization: bool = False, log_dir: str = '', comment: str = ''):
        """
        以單張圖片的格式來測試網絡
        Parameters
        ----------
        img:數據類型爲numpy.narray
            圖片數據

        Returns
        -------
            預測類別

        Examples
        -------
            img = cv2.imread('../sources/dataset/test/12-3.jpg', 0)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5)
            ])
            k = TestModel(ResNet50(ResidualBlock, 1, 5), 'model/net_21.pth').testSingleImg(transform(img).view(1, 1, 224, 224))
            print(k)
        """
        if visualization:
            from torch.utils.tensorboard import SummaryWriter
            if log_dir == '' or comment == '':
                raise RuntimeError('SummaryWriter配置空!')
            with SummaryWriter(log_dir=log_dir, comment=comment) as writer:
                # 添加输入图片
                writer.add_images('input img', img)
                # 可视化网络结构
                writer.add_graph(self.model, img)
        else:
            out = self.model(img)
            _, prediction = torch.max(out, 1)
            return prediction.numpy()[0]

    def testDataLoader(self, dataLoader: DataLoader, plot_confusion_matrix: bool = False):
        """
        以DataLoader的形式來測試網絡預測的準確度，命令行會打印出準確率
        Parameters
        ----------
        dataLoader: DataLoader
            DataLoader類型的數據集

        Examples
        -------
            transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(256, 256)),
            torchvision.transforms.ToTensor(),
            ])
            dataset = torchvision.datasets.ImageFolder('../sources/dataset/test/', transform)
            testDataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
            TestModel(ResNet18(5), './model/cnn_model/net_21.pth').testDataLoader(testDataloader)

        """
        from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
        error_num = 0
        total = 0
        y_pred = []
        y_true = []
        print('正在计算准确率...')
        for i, data in enumerate(dataLoader):
            inputs, labels = data
            if labels[0] != 0:
                l = 1
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            with torch.no_grad():
                try:
                    outputs = self.model(inputs)
                except RuntimeError as e:
                    error_num += 1
                    continue
            # TODO: if这里修改一下
            if isinstance(self.model, int):
                k = torch.gt(outputs, 0.5)
                prediction = torch.sum(k, dim=1)
                y_pred.append(prediction.cpu().numpy().astype(np.str).tolist())
                y_true.append(labels.cpu().numpy().astype(np.str).tolist())
            else:
                _, prediction = torch.max(outputs, 1)
                y_pred.append(prediction.numpy().astype(np.str).tolist())
                y_true.append(labels.numpy().astype(np.str).tolist())
            if labels.size().__len__() == 0:
                total += 1
            else:
                total += labels.size()[0]
        y_pred = np.array(functools.reduce(operator.concat, y_pred), dtype=np.int64)
        y_true = np.array(functools.reduce(operator.concat, y_true), dtype=np.int64)
        confusionMatrixDisplay(y_true, y_pred) if plot_confusion_matrix else total
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        with open('./evalution.txt', 'a+') as f:
            f.write('Total Sample:{}, accuracy:{:.3f}, precison:{:.3f}, recall:{:.3f}, f1_score:{:.3f}\n'
                    .format(total, accuracy, precision, recall, f1))
            if error_num != 0:
                f.write('有{}张图片发生错误\n'.format(error_num))

    def getFeatureVector(self, img, visualization=False, log_dir='', comment=''):
        """
        将单张图片转换成特征向量
        Parameters
        ----------
            img: 数据类型为numpy.narray
                圖片數據，數組格式爲(1, channel_num, H, W)
            visualization: bool
                是否开启可视化
            log_dir: str
                tensorboard文件存储位置
            comment: str
                tensorboard comment

        Returns
        -------
            返回圖片的特徵向量
        """
        if visualization:
            from torch.utils.tensorboard import SummaryWriter
            if log_dir == '' or comment == '':
                raise RuntimeError('SummaryWriter配置空!')
            with SummaryWriter(log_dir=log_dir, comment=comment) as writer:
                # 添加输入图片
                writer.add_image('input img', img)
                # 可视化网络结构
                writer.add_graph(self.model, img)
        else:
            return self.model(img).detach().numpy().flatten()


def getFeatureVectorPlus(img,
                         img_save_path: str,
                         res_save_path: str,
                         k: int,
                         deg: int,
                         transform,
                         model,
                         model_dict_path: str,
                         crop: list = [0, 0, 0, 0],
                         strict: bool = False,
                         n: int = 5,
                         kernel_size: int = 3,
                         progress: bool = False):
    """
    提取一张不经过任何处理图片的特征向量
    Parameters
    ----------
    img
    img_save_path
    res_save_path
    k
    deg
    transform
    model
    model_dict_path
    crop
    strict
    n
    kernel_size
    progress

    Returns
    -------
    Examples
    --------
       getFeatureVectorPlus(img=img,
                            img_save_path='test/img/te-12-262.jpg',
                            res_save_path='test/res/te-12-262.txt',
                            k=190,
                            deg=3,
                            transform=torchvision.transforms.ToTensor(),
                            model=ResNet50Regression(1),
                            model_dict_path='model/net_22.pth',
                            crop=[60, 0, 30, 30]
                            )
    """
    import cv2
    import numpy as np
    surface, _ = surfaceFitting(img, deg=deg)
    if progress:
        cv2.imshow('PRESS TO CONTINUE', _)
        cv2.waitKey(0)
    surfacePosition = np.array([img.shape[1] - i for i in surface])
    ret = flatten(img, surfacePosition)
    ret = denoise(ret, n=n, kernel_size=kernel_size)
    ret = cropImg(ret, crop[0], crop[1], crop[2], crop[3])
    cv2.imwrite(img_save_path, ret)
    feature = TestModel(model, model_dict_path, strict=strict) \
        .getFeatureVector(transform(np.resize(ret, (224, 224))).view(1, 1, 224, 224))
    np.savetxt(res_save_path, feature.reshape(1, -1), fmt='%f', delimiter=',')


def getAllFeatureVector(rootPath: str,
                        model,
                        modelLocation: str,
                        transform,
                        labelPath: str,
                        txtRootPath: str = ''):
    """
    將制定目錄下的各類圖片轉換成向量的形式，以便操作。
    特徵提取是由CNN完成
    Parameters
    ----------
        labelPath
        rootPath: Sting
            图片根目录。
            如果你的其中一张图片路径是../sources/dataset/apple/apple1.jpg，那么rootPath = '../sources/dataset/'

        model:
            网络模型。
            需要继承nn.Moudel

        modelLocation: String
            预训练后的模型参数文件地址。

        transform:
            需要对输入的图片做的预处理
            比如torchvision.transforms.ToTensor()是将图片转换成tensor格式

        txtRootPath:String
            图片向量保存的根目录。

    Examples
    -------
        model = ResNet50Regression(1)
        modelLocation = 'model/net_22.pth'
        rootPath = '../sources/dataset/'
        transform = torchvision.transforms.ToTensor()
        getAllFeatureVector(rootPath=rootPath, model=model, modelLocation=modelLocation, transform=transform)

    """
    import numpy as np
    import json
    import os
    import cv2
    temp = []
    count = 0
    names = os.listdir(rootPath)
    print('wait a minute...')
    with open(labelPath, 'r') as f:
        labelDict = json.load(f)
        # try:
        for name in names:
            if os.path.isfile(rootPath + name):
                continue
            imgNames = os.listdir(rootPath + name + '/')
            for imgName in imgNames:
                img = cv2.imread(rootPath + name + '/' + imgName, flags=0)
                # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                k = TestModel(model=model, modelLocation=modelLocation, strict=False) \
                    .getFeatureVector(transform(img).view(1, 1, 256, 256))
                k = np.append(k, labelDict[name])
                temp.append(k)
                count += 1
            print(name + ' saved successfully! number is {}'.format(count))
            count = 0
        temp = np.array(temp)
        np.savetxt(txtRootPath + 'vector' + '.txt', temp, '%f', delimiter=',')
        # except Exception as e:
        #     print('Save failed\n' + e.__str__())


def make_labels(root_path: str, save_path: str, label_location: str):
    """
    为图片特征向量最后一列中加上对应标签
    Parameters
    ----------
    root_path: String
        图片向量文件存放的根路径
        root_path/feature.txt

    save_path: String
        加上标签后文件存放的位置

    label_location: String
        label文件的路径

    Notes
    -----
    label是以json格式保存在可读文件中, 键名是图片向量的文件名， 值为标签
    比如apple这个类的图片向量被保存到apple.txt中，并且它对应的标签为1，
    label中就是 {"apple": 1}

    """
    import os
    import json
    import numpy as np
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(label_location, 'r') as f:
        s1 = json.load(f)
        for key, label in s1.items():
            path_key = root_path + key + '.txt'
            if os.path.exists(path_key):
                with open(path_key, 'r') as f1:
                    content = f1.readlines()
                    for i in range(len(content)):
                        content[i] = content[i].strip().split(',')
                    content = np.array(content).astype(np.float64)
                    content = np.insert(content, content.shape[1], s1[key], axis=1)
                    np.savetxt(save_path + key + '-done.txt', content, '%f', delimiter=',')


class MarkLabel:
    """
    OCT图像分层标记
    """

    def __init__(self, win_name: str):
        import cv2
        cv2.namedWindow(win_name, 0)
        self.win_name = win_name
        self.surfaceLoc = []
        print('标记完后按下Enter保存，退出按ESC键\n')

    def on_mouse_pick_points(self, event, x, y, flags, param):
        """
        鼠标事件回调函数
        """
        import cv2
        if flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.drawMarker(param, (x, y), 255, markerSize=1)
            self.surfaceLoc.append((512 - y, x))

    def markLabel(self, img, save_path: str = None):
        """
        分层标记点
        Parameters
        Examples
        --------
        MarkLabel('show').markLabel(image, 'test/img/262label.jpg')
        """
        import cv2
        ret = np.copy(img)
        cv2.setMouseCallback(self.win_name, self.on_mouse_pick_points, ret)
        while True:
            cv2.imshow(self.win_name, ret)
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
            elif key == 13:  # enter
                if save_path is None:
                    print('没有给定文件路径!\n')
                    continue
                cv2.imwrite(save_path, ret)
                print('保存成功\n')
        cv2.destroyAllWindows()
        return list(set(self.surfaceLoc))


def extract_ROI(img, margin: int, diff: int = 0, k: int = 1, winStep: int = 1):
    """
    使用滑动窗口来提取ROI区域
    Parameters
    ----------
    img : narray
        图片数据
    winStep : int
        窗口滑动步长
    k : int
        需要保留几个窗口
    diff : int
        窗口上下裁剪,diff越上面割的越小
    margin
    Returns
    -------
    Examples
    --------
        # 提取目录下所有图片的ROI区域
        rootPath = '../sources/dataset/preprocessed/'
        # categories = os.listdir(rootPath)
        categories = ['0']
        for category in categories:
            count = 0
            imgNames = os.listdir(rootPath + category + '/')
            if not os.path.exists(rootPath + category + '/done/'):
                os.mkdir(rootPath + category + '/done/')
            for imgName in imgNames:
                if not imgName[-3:] == 'jpg':
                    continue
                img = cv2.imread(rootPath + category + '/' + imgName, flags=0)
                margin = int((img.shape[1] - int(img.shape[0] / 2)) / 2)
                ret = extract_ROI(img, margin=margin, diff=-50, winStep=50)[0]
                cv2.imwrite(rootPath + category + '/done/' + imgName, ret)
                count += 1
                print('class: {}, total number: {}, current: {}, left number: {}'.format(category, len(imgNames),
                                                                                    imgName, len(imgNames) - count))
    """
    import numpy as np
    width = img.shape[0]
    startCol = 0
    endCol = int(width / 2)
    maxStatue = np.array([0] * 3)
    while endCol < width:
        # 滑动窗口前进
        winArea = img[:, startCol:endCol]
        croped = cropImg(winArea[:, startCol:endCol], margin + diff, margin - diff, 0, 0)
        E = calc_2D_Entropy(croped, N=2)
        maxStatue = np.row_stack((maxStatue, np.array([startCol, endCol, E])))
        startCol += winStep
        endCol += winStep
    maxStatue = np.delete(maxStatue, [0], axis=0)
    ret = []
    for i in range(k):
        idx = np.argmax(maxStatue[:, -1])
        temp = cropImg(img[:, int(maxStatue[idx][0]):int(maxStatue[idx][1])], margin + diff, margin - diff, 0, 0)
        ret.append(temp)
        maxStatue = np.delete(maxStatue, [idx], axis=0)
    return ret


def _extract_ROI(img, margin: int, diff: int = 0, k: int = 1, winStep: int = 1):
    """
    绘制项
    Parameters
    ----------
    img
    margin
    diff
    k
    winStep

    Returns
    -------

    """
    import numpy as np
    import cv2
    import random
    width = img.shape[0]
    startCol = 0
    endCol = int(width / 2)
    maxStatue = np.array([0] * 3)
    while endCol < width:
        # 滑动窗口前进
        winArea = img[:, startCol:endCol]
        croped = cropImg(winArea[:, startCol:endCol], margin + diff, margin - diff, 0, 0)
        E = calc_2D_Entropy(croped)
        maxStatue = np.row_stack((maxStatue, np.array([startCol, endCol, E])))
        startCol += winStep
        endCol += winStep
    maxStatue = np.delete(maxStatue, [0], axis=0)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(k):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        idx = np.argmax(maxStatue[:, -1])

        left_top = (int(maxStatue[idx][0]), margin + diff)
        right_top = (int(maxStatue[idx][1]), margin + diff)
        left_bottom = (int(maxStatue[idx][0]), img.shape[1] - margin + diff)
        right_bottom = (int(maxStatue[idx][1]), img.shape[1] - margin + diff)
        point_color = (r, g, b)
        thickness = 1
        lineType = 4
        cv2.line(rgb, left_top, right_top, point_color, thickness, lineType)
        cv2.line(rgb, left_top, left_bottom, point_color, thickness, lineType)
        cv2.line(rgb, left_bottom, right_bottom, point_color, thickness, lineType)
        cv2.line(rgb, right_top, right_bottom, point_color, thickness, lineType)

        cv2.imshow(maxStatue[idx][-1].__str__(), rgb)
        cv2.waitKey(0)


def calc_2D_Entropy(img, N: int = 2):
    import cv2
    # start = time.time()
    shrink = cv2.resize(img, dsize=(64, 64))
    row = shrink.shape[0]
    col = shrink.shape[1]
    area = row * col
    # towards 1 2 3 4分别为右 上 左 下
    step = towards = 1
    count = 0
    temp = []
    saveed = []
    for i in range(row):
        for j in range(col):
            m = i
            n = j
            # 对每一个像素点提取领域信息
            while step < 2 * N + 1:
                while count != 2:
                    count += 1
                    k = step
                    while k > 0:
                        if towards == 1:
                            n += 1
                            temp.append((0 if n >= col or m >= row or n < 0 or m < 0 else shrink[m][n]))
                        if towards == 2:
                            m -= 1
                            temp.append((0 if m < 0 or n < 0 or m >= row or n >= col else shrink[m][n]))
                        if towards == 3:
                            n -= 1
                            temp.append((0 if n < 0 or m < 0 or m >= row or n >= col else shrink[m][n]))
                        if towards == 4:
                            m += 1
                            temp.append((0 if m >= row or n >= col or m < 0 or n < 0 else shrink[m][n]))
                        k -= 1
                    towards = towards % 4 + 1
                count = 0
                step += 1
            for left in range(2 * N):
                n += 1
                temp.append((0 if n >= col or m >= row or n < 0 or m < 0 else shrink[m][n]))
            # 计算j值
            jValue = sum(temp) / pow((2 * N + 1), 2)
            # (i, j)元组信息保存
            saveed.append((shrink[i][j], jValue))
            # 恢复初始值
            step = towards = 1
            count = 0
            temp = []
    # 计算-E
    _E = 0
    savedSet = list(set(saveed))
    savedSet.sort(key=saveed.index)
    for item in savedSet:
        fij = saveed.count(item)
        pij = fij / area
        _E += pij * math.log2(pij)
    # end = time.time()
    return -_E


def random_run(probability: int, func, **args):
    """
    以一点的概率来执行某个函数
    Parameters
    ----------
    probability:int
        概率值
    func:builtin_function_or_method
        需要回调执行的函数
    args:dict
        func的参数
    Returns
    -------
    Examples
    -------
        args = {'center': (rows / 2, cols / 2), 'angle': k, 'scale': 1}
        M = random_run(rotationProbability, cv2.getRotationMatrix2D, **args)

    """
    import random
    list = []
    # list中放入probability个1
    for i in range(probability):
        list.append(1)
    # 剩下的位置放入0
    for x in range(100 - probability):
        list.append(0)
    # 随机抽取一个
    a = random.choice(list)
    if a == 0:
        return None
    if a == 1:
        return func(**args)



