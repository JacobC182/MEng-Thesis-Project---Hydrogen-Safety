#M-L - Overpressure estimation script
#Masters Project 2021

#Libraries
import numpy as np
import math as ma
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xlrd
import matplotlib.pyplot as plt

#Distance range
distanceRange = range(5, 300 +1, 1)
distanceRange = np.array(distanceRange)

#Importing dataset
DataBook1 = xlrd.open_workbook("ExperimentData.xls")
DataSheet1 = DataBook1.sheet_by_index(0)

featureData = []        #array for H2 conc%, Volume, Distance - feature data list (2D)
pressureData = []       #array for result/pressure data list
for i in range(2, 88 +1, 1):
    featureData.append([DataSheet1.cell_value(i, 1),DataSheet1.cell_value(i, 2),DataSheet1.cell_value(i, 3)])    #adding all feature and pressure value data into arrays from excel document
    pressureData.append(DataSheet1.cell_value(i, 4))

#Creating test and train datasets
xTrainRaw, xTestRaw, yTrain, yTest = train_test_split(featureData, pressureData, test_size=0.2, random_state=0)

#Scaling datasets
Scaler1 = StandardScaler()
Scaler1.fit(xTrainRaw)

xTrain = Scaler1.transform(xTrainRaw)
xTest = Scaler1.transform(xTestRaw)

#Creating Models
Forest1 =  RandomForestRegressor(n_jobs=-1)
Xtrees1 = ExtraTreesRegressor(n_jobs=-1)
Gboost1 = GradientBoostingRegressor()
kNN1 = KNeighborsRegressor()
SVR1 = NuSVR()
MLP1 = MLPRegressor()

#Creating parameter tuning grids
#forest AND xtrees parameter grid are the same
paramGridForest = [
            {   'n_estimators': [50,100,150,200,250,300,350,400,450,500],
                'min_samples_split': [2,3,4],
                'min_samples_leaf': [1,2,3,4]}
]

paramGridGboost = [
            {   'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                'n_estimators': [50,100,150,200,250,300,350,400,450,500],
                'min_samples_split': [2,3,4],
                'min_samples_leaf': [1,2,3,4],
                'max_depth': [1,2,3,4,5,6,7,8]}
]

paramGridkNN = [
            {   'n_neighbors': [1,2,3,4,5,6,7,8],
                'weights': ['uniform','distance'],
                'leaf_size': [10,20,30,40,50]}
]

paramGridSVR = [
            {   'nu': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
]

paramGridMLP = [
            {   'solver': ['adam','lbfgs'],
                'hidden_layer_sizes': [(5,),(10,),(15,),(5,5,),(5,10,),(5,15,),(10,5,),(10,15,),(10,10),(15,5,),(15,10,),(15,15,),(5,15,10,),(5,10,15,),(10,15,5,),(10,5,15,),(5,5,5,),(15,15,15,),(10,10,10,)]}
]

#Creating parameter searching GridSearchObjects
forestCLF = GridSearchCV(estimator=Forest1, param_grid=paramGridForest, scoring="neg_mean_squared_error", n_jobs=-1, cv=5, refit=True)
XtreesCLF = GridSearchCV(estimator=Xtrees1, param_grid=paramGridForest, scoring="neg_mean_squared_error", n_jobs=-1, cv=5, refit=True)
GBoostCLF = GridSearchCV(estimator=Gboost1, param_grid=paramGridGboost, scoring="neg_mean_squared_error", n_jobs=-1, cv=2, refit=True)
kNNCLF = GridSearchCV(estimator=kNN1, param_grid=paramGridkNN, scoring="neg_mean_squared_error", n_jobs=-1, cv=5, refit=True)
SVR_CLF = GridSearchCV(estimator=SVR1, param_grid=paramGridSVR, scoring="neg_mean_squared_error", n_jobs=-1, cv=5, refit=True)
MLP_CLF = GridSearchCV(estimator=MLP1, param_grid=paramGridMLP, scoring="neg_mean_squared_error", n_jobs=-1, cv=5, refit=True)

#Fitting grid search models for best parameters
print("---")
print("Working...")

forestCLF.fit(xTrain, yTrain)
bestForest = forestCLF.best_estimator_

XtreesCLF.fit(xTrain, yTrain)
bestXtrees = XtreesCLF.best_estimator_

GBoostCLF.fit(xTrain, yTrain)
bestGboost = GBoostCLF.best_estimator_

kNNCLF.fit(xTrain, yTrain)
bestkNN = kNNCLF.best_estimator_

SVR_CLF.fit(xTrain, yTrain)
bestSVR = SVR_CLF.best_estimator_

MLP_CLF.fit(xTrain, yTrain)
bestMLP = MLP_CLF.best_estimator_


#Printing best parameters and scores
print("---")

print("RF Best:")
print(forestCLF.best_params_)
print("RF score: " +  str(bestForest.score(xTest, yTest)))

print("XTrees Best:")
print(XtreesCLF.best_params_)
print("XTrees score: " + str(bestXtrees.score(xTest, yTest)))

print("GBoost Best:")
print(GBoostCLF.best_params_)
print("GBoost score: " + str(bestGboost.score(xTest, yTest)))

print("kNN Best:")
print(kNNCLF.best_params_)
print("kNN score: " + str(bestkNN.score(xTest, yTest)))

print("SVR Best:")
print(SVR_CLF.best_params_)
print("SVR score: " + str(bestSVR.score(xTest, yTest)))

print("MLP Best:")
print(MLP_CLF.best_params_)
print("MLP score: " + str(bestMLP.score(xTest, yTest)))

#Creating 5 random dataset splits
xTrainRaw1, xTestRaw1, yTrain1, yTest1 = train_test_split(featureData, pressureData, test_size=0.2)
xTrainRaw2, xTestRaw2, yTrain2, yTest2 = train_test_split(featureData, pressureData, test_size=0.2)
xTrainRaw3, xTestRaw3, yTrain3, yTest3 = train_test_split(featureData, pressureData, test_size=0.2)
xTrainRaw4, xTestRaw4, yTrain4, yTest4 = train_test_split(featureData, pressureData, test_size=0.2)
xTrainRaw5, xTestRaw5, yTrain5, yTest5 = train_test_split(featureData, pressureData, test_size=0.2)

splitScaler1 = StandardScaler()
splitScaler2 = StandardScaler()
splitScaler3 = StandardScaler()
splitScaler4 = StandardScaler()
splitScaler5 = StandardScaler()

splitScaler1.fit(xTrainRaw1)
xTrain1 = splitScaler1.transform(xTrainRaw1)
xTest1 = splitScaler1.transform(xTestRaw1)

splitScaler2.fit(xTrainRaw2)
xTrain2 = splitScaler2.transform(xTrainRaw2)
xTest2 = splitScaler2.transform(xTestRaw2)

splitScaler3.fit(xTrainRaw3)
xTrain3 = splitScaler3.transform(xTrainRaw3)
xTest3 = splitScaler3.transform(xTestRaw3)

splitScaler4.fit(xTrainRaw4)
xTrain4 = splitScaler4.transform(xTrainRaw4)
xTest4 = splitScaler4.transform(xTestRaw4)

splitScaler5.fit(xTrainRaw5)
xTrain5 = splitScaler5.transform(xTrainRaw5)
xTest5 = splitScaler5.transform(xTestRaw5)

scores_RF = []
scores_XT = []
scores_GB = []
scores_kN = []
scores_SV = []
scores_MP = []

bestForest.fit(xTrain1, yTrain1)
scores_RF.append(bestForest.score(xTest1, yTest1))
bestForest.fit(xTrain2, yTrain2)
scores_RF.append(bestForest.score(xTest2, yTest2))
bestForest.fit(xTrain3, yTrain3)
scores_RF.append(bestForest.score(xTest3, yTest3))
bestForest.fit(xTrain4, yTrain4)
scores_RF.append(bestForest.score(xTest4, yTest4))
bestForest.fit(xTrain5, yTrain5)
scores_RF.append(bestForest.score(xTest5, yTest5))

bestXtrees.fit(xTrain1, yTrain1)
scores_XT.append(bestXtrees.score(xTest1, yTest1))
bestXtrees.fit(xTrain2, yTrain2)
scores_XT.append(bestXtrees.score(xTest2, yTest2))
bestXtrees.fit(xTrain3, yTrain3)
scores_XT.append(bestXtrees.score(xTest3, yTest3))
bestXtrees.fit(xTrain4, yTrain4)
scores_XT.append(bestXtrees.score(xTest4, yTest4))
bestXtrees.fit(xTrain5, yTrain5)
scores_XT.append(bestXtrees.score(xTest5, yTest5))

bestGboost.fit(xTrain1, yTrain1)
scores_GB.append(bestGboost.score(xTest1, yTest1))
bestGboost.fit(xTrain2, yTrain2)
scores_GB.append(bestGboost.score(xTest2, yTest2))
bestGboost.fit(xTrain3, yTrain3)
scores_GB.append(bestGboost.score(xTest3, yTest3))
bestGboost.fit(xTrain4, yTrain4)
scores_GB.append(bestGboost.score(xTest4, yTest4))
bestGboost.fit(xTrain5, yTrain5)
scores_GB.append(bestGboost.score(xTest5, yTest5))

bestkNN.fit(xTrain1, yTrain1)
scores_kN.append(bestkNN.score(xTest1, yTest1))
bestkNN.fit(xTrain2, yTrain2)
scores_kN.append(bestkNN.score(xTest2, yTest2))
bestkNN.fit(xTrain3, yTrain3)
scores_kN.append(bestkNN.score(xTest3, yTest3))
bestkNN.fit(xTrain4, yTrain4)
scores_kN.append(bestkNN.score(xTest4, yTest4))
bestkNN.fit(xTrain5, yTrain5)
scores_kN.append(bestkNN.score(xTest5, yTest5))

bestSVR.fit(xTrain1, yTrain1)
scores_SV.append(bestSVR.score(xTest1, yTest1))
bestSVR.fit(xTrain2, yTrain2)
scores_SV.append(bestSVR.score(xTest2, yTest2))
bestSVR.fit(xTrain3, yTrain3)
scores_SV.append(bestSVR.score(xTest3, yTest3))
bestSVR.fit(xTrain4, yTrain4)
scores_SV.append(bestSVR.score(xTest4, yTest4))
bestSVR.fit(xTrain5, yTrain5)
scores_SV.append(bestSVR.score(xTest5, yTest5))

bestMLP.fit(xTrain1, yTrain1)
scores_MP.append(bestMLP.score(xTest1, yTest1))
bestMLP.fit(xTrain2, yTrain2)
scores_MP.append(bestMLP.score(xTest2, yTest2))
bestMLP.fit(xTrain3, yTrain3)
scores_MP.append(bestMLP.score(xTest3, yTest3))
bestMLP.fit(xTrain4, yTrain4)
scores_MP.append(bestMLP.score(xTest4, yTest4))
bestMLP.fit(xTrain5, yTrain5)
scores_MP.append(bestMLP.score(xTest5, yTest5))

scores_RF = np.array(scores_RF)
scores_XT = np.array(scores_XT)
scores_GB = np.array(scores_GB)
scores_kN = np.array(scores_kN)
scores_SV = np.array(scores_SV)
scores_MP = np.array(scores_MP)

print("---")

print("RF score: " + str(np.mean(scores_RF))[0:7] + "  STD-DEV " + str(np.std(scores_RF))[0:7])
print("XT score: " + str(np.mean(scores_XT))[0:7] + "  STD-DEV " + str(np.std(scores_XT))[0:7])
print("GB score: " + str(np.mean(scores_GB))[0:7] + "  STD-DEV " + str(np.std(scores_GB))[0:7])
print("kN score: " + str(np.mean(scores_kN))[0:7] + "  STD-DEV " + str(np.std(scores_kN))[0:7])
print("SV score: " + str(np.mean(scores_SV))[0:7] + "  STD-DEV " + str(np.std(scores_SV))[0:7])
print("MP score: " + str(np.mean(scores_MP))[0:7] + "  STD-DEV " + str(np.std(scores_MP))[0:7])

volume = 300
concentration = 30

volumeList = []
concList = []

for i in range(len(distanceRange)):
    volumeList.append(volume)
    concList.append(concentration)

volumeList = np.array(volumeList)
concList = np.array(concList)

ML_input = np.concatenate((concList, volumeList, distanceRange))

ML_input = np.reshape(ML_input, (3, len(distanceRange)))
ML_input = np.transpose(ML_input)


bestForest.fit(xTrain, yTrain)
bestXtrees.fit(xTrain, yTrain)
bestGboost.fit(xTrain, yTrain)
bestSVR.fit(xTrain, yTrain)
bestMLP.fit(xTrain, yTrain)

RF_results = []
XT_results = []
GB_results = []
SV_results = []
MP_results = []




for i in range(len(distanceRange)):
    RF_results.append(bestForest.predict([[concentration, volume, distanceRange[i]]])[0])
    XT_results.append(bestXtrees.predict([[concentration, volume, distanceRange[i]]])[0])
    GB_results.append(bestGboost.predict([[concentration, volume, distanceRange[i]]])[0])
    SV_results.append(bestSVR.predict([[concentration, volume, distanceRange[i]]])[0])
    MP_results.append(bestMLP.predict([[concentration, volume, distanceRange[i]]])[0])

#RF_results = 

plt.figure("Machine Learning Models - OverPressure vs. Distance Graph")

plt.plot(distanceRange, RF_results)
plt.plot(distanceRange, XT_results)
plt.plot(distanceRange, GB_results)
plt.plot(distanceRange, SV_results)

plt.grid()
plt.show()