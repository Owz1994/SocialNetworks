import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
data = pd.read_excel("训练.xlsx")  #从excel读取数据
testA = pd.read_excel("测试A.xlsx")
testB = pd.read_excel("测试B.xlsx")
#train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
#按4;1分割训练集和测试集
"""
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_sindex]
    strat_test_set = data.loc[test_index]
"""
#按照某一列属性比例分
corr_matrix = data.corr()
a=corr_matrix["Y"]
a.to_csv("res.csv")

ros = pd.read_csv("res.csv")
data_clean = pd.DataFrame(data=None, index=None, columns=None)
testA_clean = pd.DataFrame(data=None, index=None, columns=None)
testB_clean = pd.DataFrame(data=None, index=None, columns=None)
data_clean["Y"] = data["Y"]
for i in range(len(ros)):
    if ros.iloc[i].values[1] and abs(ros.iloc[i].values[1]) > 0.1 and abs(ros.iloc[i].values[1]) < 1:
        data_clean[ros.iloc[i].values[0]] = data[ros.iloc[i].values[0]]
        testA_clean[ros.iloc[i].values[0]] = testA[ros.iloc[i].values[0]]
        testB_clean[ros.iloc[i].values[0]] = testB[ros.iloc[i].values[0]]
#清楚相关系数小的列
#train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
data_feature = data_clean.drop("Y",axis = 1)
data_label = data_clean["Y"].copy()

pipline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
data_pipl = pipline.fit_transform(data_feature)
testA_pipl = pipline.fit_transform(testA_clean)
testB_pipl = pipline.fit_transform(testB_clean)


line_reg = LinearRegression()
line_reg.fit(data_pipl,data_label)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_pipl,data_label)

forest_reg = RandomForestRegressor()
forest_reg.fit(data_pipl,data_label)

param_grid = {
        "base_estimator__min_samples_split": [20],  # 分裂时，内部节点内样本不能低于 min_samples_split
        "base_estimator__min_samples_leaf": [5],  # 剪枝时，叶子节点少于该样本数会被减掉 min_samples_leaf
        "base_estimator__max_depth": [35],
        "n_estimators": [300],
        "learning_rate": [0.002]
 }

DTC = DecisionTreeRegressor(random_state=42)
ABC = AdaBoostRegressor(base_estimator=DTC,
                        loss="square")

grid_search = GridSearchCV(estimator=ABC,
                           param_grid=param_grid,
                           cv=10, scoring='neg_mean_absolute_error',
                           return_train_score=True)
grid_search.fit(data_pipl, data_label)

# model = XGBRegressor(
#     learning_rate = 0.1,
#     n_estimators = 1000,
#     max_depth = 7,
#     min_child_weight = 5,
#     subsample = 0.8,
#     colsample_bytree = 0.8,
#     seed = 0
# )
# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# param_test2 = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[4,5,6]
# }
#ABC = AdaBoostRegressor(base_estimator=XGBRegressor)
# grid_search = GridSearchCV(estimator=model,
#                            param_grid=param_test1,
#                            cv=10, scoring='neg_mean_absolute_error',
#                            return_train_score=True)
# grid_search.fit(data_pipl, data_label)
# model.fit( data_pipl, data_label, eval_metric='mae',
#     #eval_set=[(X_train, y_train), (X_valid, y_valid)],
#     early_stopping_rounds=20)
#y_pred = model.predict( X_valid )
# print(grid_search.best_params_)
#preA = grid_search.predict(testA_pipl)
#preB = grid_search.predict(testB_pipl)


# param_grid = [
#     {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
#     {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
# ]
# forest_reg = RandomForestRegressor()
# grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring = 'neg_mean_squared_error')
# grid_search.fit(data_pipl,data_label)
# print(grid_search.best_params_)

scores_line = cross_val_score(line_reg,data_pipl,data_label,scoring="neg_mean_squared_error",cv=10)
scores_tree = cross_val_score(tree_reg,data_pipl,data_label,scoring="neg_mean_squared_error",cv=10)
scores_forest = cross_val_score(forest_reg,data_pipl,data_label,scoring="neg_mean_squared_error",cv=10)
scores_grid = cross_val_score(grid_search,data_pipl,data_label,scoring="neg_mean_squared_error",cv=10)
print(np.sqrt(-scores_line),np.sqrt(-scores_tree),np.sqrt(-scores_forest),np.sqrt(-scores_grid))
preA = grid_search.predict(testA_pipl)
preB = grid_search.predict(testB_pipl)
IDA = testA["ID"]
IDB = testB["ID"]

resA = pd.DataFrame([IDA,preA],index=None, columns=None).T
resB = pd.DataFrame([IDB,preB],index=None, columns=None).T
#lin_mse = mean_squared_error(test_label,pre)


resA.to_csv("测试A-答案模板.csv",index=False,columns = None)
resB.to_csv("测试B-答案模板.csv",index=False,columns = None)




