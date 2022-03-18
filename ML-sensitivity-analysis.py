#machine learning sensitivity analysis script

#importing libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import xlrd, xlwt

#Distance range
distanceRange = range(2, 100 +1, 1)
distanceRange = np.array(distanceRange)

#importing dataset
DataBook1 = xlrd.open_workbook("ExperimentData.xls")
DataSheet1 = DataBook1.sheet_by_index(0)

featureData = []        #array for H2 conc%, Volume, Distance - feature data list (2D)
pressureData = []       #array for result/pressure data list
for i in range(2, 88 +1, 1):
    featureData.append([DataSheet1.cell_value(i, 1),DataSheet1.cell_value(i, 2),DataSheet1.cell_value(i, 3)])    #adding all feature and pressure value data into arrays from excel document
    pressureData.append(DataSheet1.cell_value(i, 4))

def MLPgrid():

    def powerset(fullset):
        listsub = list(fullset)
        subsets = []
        for i in range(2**len(listsub)):
            subset = []
            for k in range(len(listsub)):            
                if i & 1<<k:
                    subset.append(listsub[k])
            subsets.append(subset)        
        return subsets

    subsets = powerset(set([2,3,4,5,6,7,8]))
    subsets2 = subsets

    for subset in subsets:
        if len(subset) > 4:
            subsets2.remove(subset)

    subsets2.pop(0)
    from itertools import permutations
    subsets3 = []

    for subset in subsets2:
        for x in permutations(subset):
            subsets3.append(x)

    return subsets3
layerGrid = MLPgrid()

#hyperparameter grid search values
paramGridForest = [
            {   'n_estimators': [*range(10,300+1,5)],
                'min_samples_split': [2],
                'min_samples_leaf': [1]}
]
paramGridGboost = [
            {   'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                'n_estimators': [*range(10,300+1,10)],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_depth': [1,2,3,4,5,6,7,8]}
]

paramGridkNN = [
            {   'n_neighbors': [*range(1,10+1,1)],
                'weights': ['uniform','distance'],
                'leaf_size': [*range(5,50+1,1)]}
]

paramGridSVR = [
            {   'nu': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                'C': [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5],
                'kernel': ["linear","poly","rbf","sigmoid"],
                'degree': [2,3,4,5]}
]

paramGridMLP = [
            {   'solver': ['lbfgs'],
                'hidden_layer_sizes': layerGrid,
                'activation': ["relu","tanh","logistic"]}
]
#Create and fit models
for i in range(1):
    Forest = GridSearchCV(estimator= RandomForestRegressor(), param_grid= paramGridForest, scoring="neg_mean_squared_error", n_jobs=-1, cv=10, refit=True).fit(featureData, pressureData)
    print(Forest.best_params_)
    print(Forest.best_score_)

    Forest = GridSearchCV(estimator= ExtraTreesRegressor(), param_grid= paramGridForest, scoring="neg_mean_squared_error", n_jobs=-1, cv=10, refit=True).fit(featureData, pressureData)
    print(Forest.best_params_)
    print(Forest.best_score_)

    Forest = GridSearchCV(estimator= GradientBoostingRegressor(), param_grid= paramGridGboost, scoring="neg_mean_squared_error", n_jobs=-1, cv=10, refit=True).fit(featureData, pressureData)
    print(Forest.best_params_)
    print(Forest.best_score_)

    Forest = GridSearchCV(estimator= KNeighborsRegressor(), param_grid= paramGridkNN, scoring="neg_mean_squared_error", n_jobs=-1, cv=10, refit=True).fit(featureData, pressureData)
    print(Forest.best_params_)
    print(Forest.best_score_)

    Forest = GridSearchCV(estimator= NuSVR(), param_grid= paramGridSVR, scoring="neg_mean_squared_error", n_jobs=-1, cv=10, refit=True).fit(featureData, pressureData)
    print(Forest.best_params_)
    print(Forest.best_score_)

    Forest = GridSearchCV(estimator= MLPRegressor(), param_grid= paramGridMLP, scoring="neg_mean_squared_error", n_jobs=-1, cv=10, refit=True).fit(featureData, pressureData)
    print(Forest.best_params_)
    print(Forest.best_score_)