#ML pressure predictions script

#importing libraries
from time import time
#from matplotlib.lines import _LineStyle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
import xlrd
import matplotlib.pyplot as plt

#Distance range
distanceRange = range(2, 100 +1, 1)
distanceRange = np.array(distanceRange)

#Importing dataset
DataBook1 = xlrd.open_workbook("ExperimentData.xls")
DataSheet1 = DataBook1.sheet_by_index(0)

featureData = []        #array for H2 conc%, Volume, Distance - feature data list (2D)
pressureData = []       #array for result/pressure data list
for i in range(2, 88 +1, 1):
    featureData.append([DataSheet1.cell_value(i, 1),DataSheet1.cell_value(i, 2),DataSheet1.cell_value(i, 3)])    #adding all feature and pressure value data into arrays from excel document
    pressureData.append(DataSheet1.cell_value(i, 4))

distanceData = []
for i in range(2, 13+1, 1):
    distanceData.append(DataSheet1.cell_value(i,3))

Scaler1 = StandardScaler()
Scaler1.fit(featureData)

featureDataScaled = Scaler1.transform(featureData)

RandomForest = RandomForestRegressor(n_estimators=350, n_jobs=-1).fit(featureData, pressureData)
ExtraTrees = ExtraTreesRegressor(n_estimators=350, n_jobs=-1).fit(featureData, pressureData)
GBoost = GradientBoostingRegressor(learning_rate=0.8, n_estimators=350).fit(featureData, pressureData)
SVR = NuSVR(nu=1).fit(featureData, pressureData)
kNN = KNeighborsRegressor(n_neighbors=2).fit(featureData, pressureData)
MLP = MLPRegressor(hidden_layer_sizes=(3,6,3), solver="lbfgs").fit(featureDataScaled, pressureData)

CloudSize = 375
Concentration = 5.1

RF_results = []
XT_results = []
GB_results = []
SV_results = []
KN_results = []
MP_results = []

for distance in distanceRange:
    RF_results.append(RandomForest.predict([[Concentration, CloudSize, distance]])[0])
    XT_results.append(ExtraTrees.predict([[Concentration, CloudSize, distance]])[0])
    GB_results.append(GBoost.predict([[Concentration, CloudSize, distance]])[0])
    SV_results.append(SVR.predict([[Concentration, CloudSize, distance]])[0])
    KN_results.append(kNN.predict([[Concentration, CloudSize, distance]])[0])
    MP_results.append( MLP.predict(Scaler1.transform([[Concentration, CloudSize, distance]]))[0])

plt.figure("M-L OverPressure Prediction Results")
plt.xlabel("Distance (m)")
plt.ylabel("Blast OverPressure (kPa)")

#plt.plot(distanceRange, RF_results)
plt.plot(distanceRange, XT_results)
#plt.plot(distanceRange, GB_results, linestyle="dotted")
#plt.plot(distanceRange, SV_results, linestyle="dashed")
#plt.plot(distanceRange, KN_results, linestyle="dotted")
#plt.plot(distanceRange, MP_results, linestyle="dashed")

#plt.scatter(distanceData[0:9], pressureData[0:9])

plt.legend(["Random Forest", "ExtraTrees", "GradientBoost", "Support Vector", "k-Neighbours", "Perceptron", "Nearest Experiment Data"])
plt.legend(["ExtremelyRandomised Trees"])
plt.grid()
plt.show()