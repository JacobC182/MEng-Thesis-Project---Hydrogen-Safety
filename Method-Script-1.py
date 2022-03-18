#################################################
#-Written by: Jacob Currie - 8th September 2021-#
#------------------------------------------------
#Hydrogen safe distance estimation methods script
#Masters project 2021 - Strathclyde-TUAS-Elomatic
#------------------------------------------------
#INSTRUCTIONS FOR USE:
#Inputs for Model - 
#1 - Container Pressure (Pa)
#2 - Cloud Volume       (m3)
#3 - Average H2 Volume Fraction  - (Range 0 - 1)
#------------------------------------------------
#OPTIONS: ("Yes"/"No")
#Enable Pressure Plotting
Plot_Pressure = "Yes"
#Enable Pressure Plotting
Plot_HeatFlux = "Yes"
#Run Monte-Carlo Sensivity Analysis
MonteCarlo_On = "Yes"
#Save Results to Excel file
Save_To_Excel = "No"
#Print Specific Distance results to console
Print_Limits_Console = "No"
#Lower Distance limit for curve evaluation
Starting_Distance = 1       #A value of <1 is not recommended
#Upper Distance limit for curve evaluation
End_Distance = 120
#Resolution for distance grid
Distance_Step = 0.5
#Libraries - [DEPENDANCIES]
import math as ma               #Math library
import matplotlib.pyplot as plt #Plotting "matplotlib" library
import numpy as np              #NumPy library
import openpyxl                 #excel ".xlsx" modern excel library
import xlrd                     #excel sheet reading library "xlrd"
import xlwt                     #excel sheet writing library "xlwt"
from sklearn.ensemble import ExtraTreesRegressor  #Extremely Randomised Trees Model object from Scikit-Learn library
import time                     #Timer library
from scipy.optimize import curve_fit    #Conventional non-linear curve fitting function from SciPy - used in BST method
from os.path import exists      #Method for checking if  file exists - used to prevent overwriting of files!

#------------------------------------------------
#Timer start
startTime = time.time()

#------------------------------------------------
#Inputs - Enter Here
P1 = 35    *100000    #Container Pressure -> (Pa)
V1 = 66              #Cloud Volume  -> (m3)
H1 = 0.06815          #Concentration of H2 in the cloud (Volume % /100)

#------------------------------------------------
#Global Variables
DistanceList = list(np.linspace(start=Starting_Distance, stop=End_Distance, num=int(np.round((End_Distance-Starting_Distance)/Distance_Step)), endpoint=True))#Distance range
Hd = 0.0841             #H2 Density at 1 ATM and 15 degrees C (kg/m3)
M1 = Hd * V1 * H1       #Mass of Hydrogen in Cloud (kg) FOR HEAT FLUX METHODS- Mass of cloud using above density relation

P_limit = [15,10,5]  #Pressure limits (kPa)
H_limit = [8,3,1.5]     #Heat Flux limits (kW/m2)
#------------------------------------------------
#Useful Functions
def IdealGasExpansionEnergy(Pi: float, Vi:float) -> float:    #Calculate the energy released from the expansion of H2 via Ideal gas assumption
    y = 1.41    #specific heat ratio of H2
    Pamb = 101325   #Ambient Pressure (Pa)

    Energy = (Pi - Pamb)*Vi/(y-1)   #Energy from gas expansion equation (ideal)
    return Energy

def Pressure(Ps: float) -> float:   #Convert scaled overpressure to pressure relative to ambient (Pa)
    Pamb = 101325   #Ambient pressure (Pa)
    P = Ps * Pamb   #Calculate pressure
    return P

def BSTcurveRoutine():  #Sub-routine that returns curve-fit for the scaled distance/scaled pressure blast curve for BST method (Mach No. = 0.35)
    #Scaled Distance and Pressure Dataset for M = 0.35, taken from online graphs!!
    dataRscaled = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #scaled distance data
    dataPscaled = [0.21, 0.21, 0.21, 0.18, 0.17, 0.15, 0.12, 0.11, 0.1, 0.09, 0.04, 0.03, 0.02, 0.018, 0.017, 0.015, 0.013, 0.01, 0.009]    #scaled overpressure data

    def f1(x, a, b):    #Simple power curve equation (y = a*x^b)
        return a*(x**b)

    BSTcurveModel = curve_fit(f1, dataRscaled, dataPscaled)

    return BSTcurveModel[0]
    
BSTcurveModel = BSTcurveRoutine()   #Calculating power curve coefficients for BST curve - [1x2] array of A,B

def BSTcurveFit(R: float) -> float:  #Curve Fitting subroutine for BST blast curves (scaled distance - R)
    
    def f1(x, a, b):    #Simple power curve equation (y = a*x^b)
        return a*(x**b)

    Ps = f1(R, BSTcurveModel[0], BSTcurveModel[1])   #estimating scaled overpressure from curve fit model - given scaled distance
    return Ps   #returning scaled overpressure

#------------------------------------------------
#Importing Experimental data set
def DataImport():
    expWB = xlrd.open_workbook("ExperimentData.xls")   #opening experiment dataset excel workbook
    expSheet = expWB.sheet_by_index(0)  #extracting experiment data sheet (sheet #1)
        
    featureData = []        #array for H2 conc%, Volume, Distance - feature data list (2D)
    pressureData = []       #array for result/pressure data list
    for i in range(2, 88 +1, 1):
        featureData.append([expSheet.cell_value(i, 1),expSheet.cell_value(i, 2),expSheet.cell_value(i, 3)])    #adding all feature and pressure value data into arrays from excel document
        pressureData.append(expSheet.cell_value(i, 4))
        #FeatureData - array of Volume[0], Concentration[1], and Distance Values[2] - 3xN list
    return featureData, pressureData

featureData, pressureData = DataImport()
#------------------------------------------------
#---------------PRESSURE METHODS-----------------
#------------------------------------------------
#Method 1 - Curve fitting method
def CurveFit(Conc, Vol, distance):  #Concentration, Cloud Volume, Target Distance - Returns Overpressure value

    Pamb = 101325   #Ambient Pressure (Pa)

    def ScaledDistance(E, Pa, R):   #Energy(E/J), Ambient pressure(Pa/Pa), Distance(R/m)
        Z = distance/((E/Pa)**(1/3))    #scaled distance formula
        return Z
    
    def Energy(V, C):   #Volume(V/m3), Concentration(C/Vol%)
        Hc = 10.79*(10**6)  #Heat of Combustion - H2 (J/m3)
        E = V * C * Hc  #Energy formula
        return E

    def ScaledOverPressure(Z, V, C):    #Distance(Z/m), Volume(V/m3), Concentration(C/Vol%)     - Curve Fitted Overpressure Equation

        A = 0.0000006948*((V*C)**3)-0.0000807*((V*C)**2)+0.002943*(V*C)+0.02095
        B = -0.7072         #A and B are substitute values/coefficients
        
        Pscaled = ( A * (Z**B) )   #Power Curve form of Scaled overpressure formula
        return Pscaled

    #Method
    ScaledPressure = ScaledOverPressure(ScaledDistance(Energy(Vol, Conc), Pamb, DistanceList), V1, H1)
    Pressure = ScaledPressure*(Pamb/1000)
    return Pressure

#------------------------------------------------
#Method 2 - TNT Equivalency method
def TNT(Vol, distance):     #Volume, Target Distance
    n = 0.1 #Explosive yield effiency (0.01-0.1 for VCE's typically) #Assumed to be 0.1 (Worst case scenario)

    massH2 = Vol*Hd #H2 cloud mass (kg)

    def equivTNTmass(mH2, n):   #mass of H2(kg), yield efficiency (0.1)

        delHc = 130800  #lower heat of combustion for H2 (kJ/kg)
        delhTNT = 4520  #lower heat of combustion for TNT (kJ/kg)

        mTNT = (mH2*delHc*n)/delhTNT    #equivalent tnt mass formula
        return mTNT
    
    def TNTsafePressure(mTNT, Z):   #(equivalent TNT mass (mTNT/kg), desired distance(Z/m))
        a = 0.3967  #coefficients of equation
        b = 3.5031
        c = 0.7241
        d = 0.0398
        Pressure = ( ma.e**((0.7241-(0.1592*ma.log(2.52079*Z)-(0.1592/3)*ma.log(mTNT)-0.03337271)**0.5)/0.0796) )  #TNT equivalency safe distance/overpressure formula
        return Pressure

    #Method
    Pressure = TNTsafePressure(equivTNTmass(massH2, n), distance)
    PressurekPa = Pressure*6.894757
    return PressurekPa  #returning pressure value

#------------------------------------------------
#Method 3 - TNO - Multi Energy method
def TNO(m1, r):    #Mass of H2 cloud (kg), distance (m)

    def PressureTNOLookup(R):
        #TNO - Multi Energy Method - Reference Curves
        #POWER CURVE ANALYTICAL FORM - LOOKUP TABLE
        def PowerCurve(a, b, R):   #Fitted Curve equation (coefficents from table - a & b) 
                            #For scaled distance (R)

            P = a* (R**b)
            return P

        #Coefficient matrix for explosion level 1, 4, 5, 6. (index-0,1,2,3)
                                #sub-index = [a, b]
        M1 = [ [10**-2, 0], [10**-1, 0], [2*(10**-1), 0], [5*(10**-1), 0] ]
        M2 = [ [6.4*(10**-3), -0.97], [6.44*(10**-2), -0.99], [0.117, -0.99], [0.301, -1.11] ]
        #Limits matrix for explosion level 1, 4, 5 ,6. (index-0,1,2,3)
                                #sub-index (lower R, upper R)
        R1 = [ [0.23, 0.6], [0.23, 0.5], [0.23, 0.6], [0.23, 0.6] ]
        R2 = [ [0.6, 7], [0.5, 70], [0.6, 90], [0.6, 100] ]

        ScaledP = []

        for i in range(4): #For all 4 explosion levels
            
            if R >= R1[i][0] and R < R1[i][1]:  #Scaled Distance Limit 1
                ScaledP.append( PowerCurve(M1[i][0], M1[i][1], R) ) #Add pressure result to output

            elif R >= R2[i][0] and R <= R2[i][1]:   #Scaled Distance Limit 2
                ScaledP.append( PowerCurve(M2[i][0], M2[i][1], R) ) #Add pressure result to output
            
            else:
                ScaledP.append(0)

        return ScaledP  #return list of 4 scaled overpressures for explosion levels

    def ScaledDistance(r, E):   #calculate scaled distance from distance(r/m) and available energy(E/J)
        Pamb = 101325   #ambient pressure (Pa)

        rd = r/((E/Pamb)**(1/3))
        return rd   #return scaled distance
    
    def Energy(m):  #calculate available energy from mass (kg)
        es = 130*(10**6)    #specific energy of H2 at 1atm/25C (J/kg)
        E = es * m
        return E*2    #return Energy (J)
    
    #Method
    R = ScaledDistance(r, Energy(m1))   #Calculating scaled distance
    Ps = PressureTNOLookup(R)           #Calculating TNO scaled pressure result from Curves/Lookup Graph
    

    P = []  #pressure list
    for scaledPressure in Ps:   #converting scaled pressure to pressure (Pa)
        P.append(Pressure(scaledPressure))

    return P #Return final Pressure

#------------------------------------------------
#Method 4 - B.S.T. method
def BST(Pinitial, Vc, R):  #Initial vessel pressure (Pinitial, Pa) - Cloud volume (Vc, m3) - Distance(R, m)
    
    Pamb = 101325   #Ambient pressure (Pa)

    #Converting cloud volume (ideal gas isothermal compression) to volume inside container
    Vcontainer = Vc * (Pamb/Pinitial)

    Elib = IdealGasExpansionEnergy(Pinitial, Vcontainer)    #liberated energy from pressure vessel rupture

    Asc = 2 #ground effect factor (Ground burst effect - reflected wave = double energy - worst case scenario)
    E = Elib * Asc  #effective energy compensating for ground effect

    Rs = R * ((Pamb/E)**(1/3))  #Calculate scaled distance from distance and energy

    Ps = BSTcurveFit(Rs)    #Estimate scaled overpressure from curve M=0.35
    return Ps*(Pamb/1000)  #Return real Overpressure

#------------------------------------------------
#Method 5 - Machine Learning model
def CreateModel():  #Sub-routine for the creation and training of ML model - creates trained model Object     
    #Configured according to the best performing model found from tuning & validation!

    #The model used = Extremely randomised trees (Variant of Random Forest)
    #Model is fitted to entire dataset in this case!
    #  - 10-fold cross validation with 90% of dataset used for training shows reliable results
    PressurePredictor = ExtraTreesRegressor(n_estimators=290, n_jobs=-1).fit(featureData, pressureData)
    return PressurePredictor    #Returning trained Model object - callable, for estimation results

#------------------------------------------------
#---------------THERMAL METHODS------------------
#------------------------------------------------
#Method 1 - lawrence J. Marchetti - (Heat flux from BLEVE Method)
def BLEVEFlux(m1,R):    #Mass of cloud, Target Distance

    d1 = 0.0841 #density of H2 in atmosphere conditions

    Vc = d1 * m1    #calculate voluume of h2 cloud
    Zp = 12.73*((Vc)**(1/3))    #calculate flame hieght
    
    d = ma.sqrt((Zp**2 + R**2)) #distance from target to centre of fireball, using pythagorus(height and horizontal distance)

    Q = (828*(m1**0.771))/(d**2)    #calculating heat flux at distance
    return Q    #returning heat flux at target distance

#------------------------------------------------
#Method 2 - Ustomin-Paltrinieri Method (Solid Flame Method)
def SolidFlame(m1 ,R):     #Mass of cloud, Target Distance

    #Equation below is taken from - The Roberts Method
    D = 7.93*(m1**(1/3))    #ma fireball diameter (m) from fuel mass (kg)

    tm = 0.45*(m1**(1/3))    #Momentum dominated fireball duration
    tb = 2.6*(m1**(1/3))     #Buoyancy dominated fireball duration

    #Assumptions - height of fireball == diameter
    v = D
    theta = 0       #Assume maximum view factor angle = maximum radiation absorption
    L = ma.sqrt((R**2 + v**2))  #calculating distance from fireball centre to target (pythagorus)
    T = 20  #ATMOSPHERE TEMPERATURE (DEGREES C)

    F = (((D/2)/L)**2)*ma.cos(theta)    #View Factor

    Tf = 2000    #Flame temperature (K)
    e = 1   #emissivity (black body = 1 (e=0:1))
    E = e * (5.67*(10**-8)) * (Tf**4)    #surface emissive power, (stefan-boltzmann constant)
    
    #Buck equation - approximation of partial pressure of water vapour in air (NOT WORKING)
    #pw = ((0.61121*ma.e)**((18.678-(T/234.5))*(T/257.14+T)))  *1000   #(Pa)
    #print(pw)
    pw = 1705   #partial vapour water pressure (Pa) - van den Bosch & Weterings 2005 - 1705Pa at 15C and Atmospheric Pressure 1ATM
    tau = 2.02*((pw*(L-(D/2)))**-0.09)     #Atmospheric attenuation/transmissivity formula

    q = tau * F * E     #Heat Flux  (W/m2)
    return(q/1000)  #return heat flux in kW/m2

#------------------------------------------------
MLpressure = CreateModel()      #Creating trained ML model method , callable Object to predict pressures - returns pressure - SYNTAX: Model.predict([Conc%,Vol,Distance])
#------------------------------------------------
#Calculating OVERPRESSURE     
CurveFitOverPressure = []                   #Empty lists to hold overpressure results for each method
TNOOverPressure = []                        
BSTOverPressure = []                        
equivTNTOverPressure = []                   
MLOverPressure = []

for distance in DistanceList:       #Iterating through every distance in Distance list and solving for Overpressure for each method
    CurveFitOverPressure.append(CurveFit(H1, V1, distance))
    TNOOverPressure.append(TNO(M1, distance))
    BSTOverPressure.append(BST(P1, V1, distance))
    equivTNTOverPressure.append(TNT(V1, distance))
    MLOverPressure.append(MLpressure.predict([[H1*100, V1, distance]])[0])  # "[0]"- Taking first/only value in returned list - ML model always returns list

#---splitting TNO explosion levels
L1 = []
L4 = []
L5 = []
L6 = []
for level in TNOOverPressure:   #Splitting TNO levels into separate lists (easier to plot)
    L1.append(level[0]/1000)    #converting from Pa to kPa
    L4.append(level[1]/1000)
    L5.append(level[2]/1000)
    L6.append(level[3]/1000)

#------------------------------------------------
#Calculating HEAT FLUX
SolidFlux = []
BleveFlux = []

for distance in DistanceList:       #Iterating through every distance in Distance list and solving for heat flux for each distance & method
    SolidFlux.append(SolidFlame(M1, distance))
    BleveFlux.append(BLEVEFlux(M1, distance))

#------------------------------------------------
#Plotting OVERPRESSURE - encapsulated plotting routine
def PlotPressure():
    #Plotting Curve fit method
    plt.figure("OverPressure vs. Distance Results")
    plt.plot(DistanceList, CurveFitOverPressure)

    #Plotting TNT method
    plt.plot(DistanceList, equivTNTOverPressure)

    #Plotting TNO Multi Energy method    
    plt.plot(DistanceList, L1)
    plt.plot(DistanceList, L4)
    plt.plot(DistanceList, L5)
    plt.plot(DistanceList, L6)

    #Plotting BST M=0.35 method
    plt.plot(DistanceList, BSTOverPressure)

    #Plotting ML method
    plt.plot(DistanceList, MLOverPressure)

    #Axis Labels
    plt.xlabel("Distance (m)")
    plt.ylabel("Overpressure (kPa)")

    #Plotting Experimental data points (distance & overpressure)
    experimentDistance = []
    for distance in featureData:
        experimentDistance.append(distance[2])

    #Plot Legend
    plt.legend(['Curve Fitting', 'TNT Equivalency', 'TNO - 1', 'TNO - 4', 'TNO - 5', 'TNO - 6', 'B.S.T.', 'ExtraTrees'])

    #PLotting save limit lines for easy graph reading
    for limit in P_limit:
        plt.plot(DistanceList, (np.zeros(len(DistanceList))+limit), linestyle="dashed")

    #Gridlines/Numbers
    plt.grid()

if Plot_Pressure.casefold() == "yes":
    PlotPressure()  #Calling Plotting Pressure routine
#------------------------------------------------
#Plotting HEAT FLUX
def PlotFlux():
    #PLotting solid flame Method
    plt.figure("Heat Flux vs. Distance")
    plt.plot(DistanceList, SolidFlux)
    #Plotting bleve Method
    plt.plot(DistanceList, BleveFlux)

    #Plot Legend
    plt.legend(['Solid Flame', 'BLEVE Heat Flux'])

    #Plot safe limit lines
    for limit in H_limit:
        plt.plot(DistanceList, (np.zeros(len(DistanceList))+limit), linestyle="dashed")

    #Axis Labels
    plt.xlabel("Distance (m)")
    plt.ylabel("Heat Flux (kW/m2)")

    plt.grid()  #Turn on gridline/units

if Plot_HeatFlux.casefold() == "yes":
    PlotFlux()  #calling PLotting Heat Flux routine
#------------------------------------------------
#EXCEL results export function
def excelSave():

    xldoc = xlwt.Workbook()     #create xl document/workbook
    xlpage1 = xldoc.add_sheet("Results")    #add sheet #1 to xl document #1

    #Writing top headings
    xlpage1.write(0,2, "Pressure")
    xlpage1.write(0,16, "Heat")

    #Writing headings to excel document
    xlpage1.write(1,1, "Distance")
    xlpage1.write(1,2, "Curve-Fit")
    xlpage1.write(1,3, "TNT-eq")
    xlpage1.write(1,4, "TNO - 1")
    xlpage1.write(1,5, "TNO - 4")
    xlpage1.write(1,6, "TNO - 5")
    xlpage1.write(1,7, "TNO - 6")
    xlpage1.write(1,8, "B.S.T")
    xlpage1.write(1,9, "ExtraTrees")
    xlpage1.write(1,16, "Solid Flame")
    xlpage1.write(1,17, "BLEVE Flux")

    #Writing input case data to excel document
    xlpage1.write(0,0, "Results")
    xlpage1.write(2,0, "Cloud Mass(kg)")
    xlpage1.write(3,0, M1)
    xlpage1.write(5,0, "Cloud Vol(m3)")
    xlpage1.write(6,0, V1)
    xlpage1.write(8,0, "H2 Conc%")
    xlpage1.write(9,0, H1*100)

    for i in range(len(DistanceList)):
        xlpage1.write(i+2,1, DistanceList[i])
        xlpage1.write(i+2,2, CurveFitOverPressure[i])
        xlpage1.write(i+2,3, equivTNTOverPressure[i])
        xlpage1.write(i+2,4, L1[i])
        xlpage1.write(i+2,5, L4[i])
        xlpage1.write(i+2,6, L5[i])
        xlpage1.write(i+2,7, L6[i])
        xlpage1.write(i+2,8, BSTOverPressure[i])
        xlpage1.write(i+2,9, MLOverPressure[i])

        xlpage1.write(i+2,16, SolidFlux[i])
        xlpage1.write(i+2,17, BleveFlux[i])

    #File overwrite checking - increasing file numbering to prevent overwriting!
    fileNum = 1
    while exists("DataExport" + str(fileNum) + ".xls"):
        fileNum += 1        

    xldoc.save("DataExport" + str(fileNum) + ".xls")    #Saving excel file

    print("Results Saved...")

if Save_To_Excel.casefold() == "yes":
    excelSave() #Calling excel saving routine

#------------------------------------------------
#MONTE-CARLO sensitivity analysis function
#Uses the distance range defined at start
#Other controls for analysis inside function!
def MonteCarlo():

    n_Samples = 1000        #Number of samples to use to run sensitivity analysis

    Volume = 170    #cloud volume (m3)
    Fraction = 0.06042  #Volume fraction of H2 (0>H2>1)

    MaxSpreadCloud = 0.2     #Maximum +- % variability of Cloud Size (0<x<1)
    MaxSpreadFraction = 0.2     #Maximum +- % variability of Volume Fraction (0<x<1)

    #Generating normal distribution array of cloud volumes
    VolumeDistribution = np.random.normal(loc=Volume, scale=MaxSpreadCloud*Volume, size=n_Samples)
    #Generating uniform distribution array of H2 volume fractions
    H2Distribution = np.random.uniform(low=Fraction-Fraction*MaxSpreadFraction, high=Fraction+Fraction*MaxSpreadFraction, size=n_Samples)

    CurveFitPressureArray = np.zeros((len(DistanceList), n_Samples))    #Creating empty 2D arrays for holding pressure curves
    TNTPressureArray = np.zeros((len(DistanceList), n_Samples))
    TNOPressureArray1 = np.zeros((len(DistanceList), n_Samples))
    TNOPressureArray4 = np.zeros((len(DistanceList), n_Samples))
    TNOPressureArray5 = np.zeros((len(DistanceList), n_Samples))
    TNOPressureArray6 = np.zeros((len(DistanceList), n_Samples))
    BSTPressureArray = np.zeros((len(DistanceList), n_Samples))
    MLPressureArray = np.zeros((len(DistanceList), n_Samples))

    for volumeCounter in range(n_Samples):
        for distanceCounter in range(len(DistanceList)):
                #Calculating pressure curves iteratively for each sample and for every distance - for every method below!
            CurveFitPressureArray[distanceCounter, volumeCounter] = CurveFit(H2Distribution[volumeCounter], VolumeDistribution[volumeCounter], DistanceList[distanceCounter])
            TNTPressureArray[distanceCounter, volumeCounter] = TNT(VolumeDistribution[volumeCounter], DistanceList[distanceCounter])
            TNOPressureArray1[distanceCounter, volumeCounter] = TNO(VolumeDistribution[volumeCounter], DistanceList[distanceCounter])[0] #TNO curve index (1,4,5,6) - [0,1,2,3]
            TNOPressureArray4[distanceCounter, volumeCounter] = TNO(VolumeDistribution[volumeCounter], DistanceList[distanceCounter])[1]
            TNOPressureArray5[distanceCounter, volumeCounter] = TNO(VolumeDistribution[volumeCounter], DistanceList[distanceCounter])[2]
            TNOPressureArray6[distanceCounter, volumeCounter] = TNO(VolumeDistribution[volumeCounter], DistanceList[distanceCounter])[3]
            BSTPressureArray[distanceCounter, volumeCounter] = BST(P1, VolumeDistribution[volumeCounter], DistanceList[distanceCounter])
            MLPressureArray[distanceCounter, volumeCounter] = MLpressure.predict([H2Distribution[volumeCounter]*100, VolumeDistribution[volumeCounter], DistanceList[distanceCounter]])[0]

    #Plotting Monte-Carlo Results - Setting up figure and graph layout!
    MCfigure = plt.figure("Monte-Carlo Results")
    CurvefitPlot = MCfigure.add_subplot(2,4,1)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("Curve Fitting Method")

    TNTPlot = MCfigure.add_subplot(2,4,2)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("TNT Equivalency Method")

    TNOPlot1 = MCfigure.add_subplot(2,4,3)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("TNO [1] Multi-Energy Method")

    TNOPlot4 = MCfigure.add_subplot(2,4,4)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("TNO [4] Multi-Energy Method")

    TNOPlot5 = MCfigure.add_subplot(2,4,5)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("TNO [5] Multi-Energy Method")

    TNOPlot6 = MCfigure.add_subplot(2,4,6)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("TNO [6] Multi-Energy Method")
    
    BSTPlot = MCfigure.add_subplot(2,4,7)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance (m)")
    plt.grid()
    plt.title("B.S.T. Method")

    MLPlot = MCfigure.add_subplot(2,4,8)
    plt.ylabel("OverPressure (kPa)")
    plt.xlabel("Distance(m")
    plt.grid()
    plt.title("Machine Learning Method")

    for i in range(n_Samples):      #Plotting result to graphs for sensitivity analysis
        CurvefitPlot.plot(DistanceList, CurveFitPressureArray[:, i])
        TNTPlot.plot(DistanceList, TNTPressureArray[:, i])
        TNOPlot1.plot(DistanceList, TNOPressureArray1[:, i]/1000)
        TNOPlot4.plot(DistanceList, TNOPressureArray4[:, i]/1000)
        TNOPlot5.plot(DistanceList, TNOPressureArray5[:, i]/1000)
        TNOPlot6.plot(DistanceList, TNOPressureArray6[:, i]/1000)
        BSTPlot.plot(DistanceList, BSTPressureArray[:, i])

    def ImportantPlot():    #Sub-function for creating a plot of the most important values from the sensitivity analysis

        MCfigure = plt.figure("Important MC-Results")   #Creating graph/figure layout - setup
        CurvefitPlot = MCfigure.add_subplot(2,4,1)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("Curve Fitting Method")

        TNTPlot = MCfigure.add_subplot(2,4,2)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("TNT Equivalency Method")

        TNOPlot1 = MCfigure.add_subplot(2,4,3)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("TNO [1] Multi-Energy Method")

        TNOPlot4 = MCfigure.add_subplot(2,4,4)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("TNO [4] Multi-Energy Method")

        TNOPlot5 = MCfigure.add_subplot(2,4,5)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("TNO [5] Multi-Energy Method")

        TNOPlot6 = MCfigure.add_subplot(2,4,6)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("TNO [6] Multi-Energy Method")
        
        BSTPlot = MCfigure.add_subplot(2,4,7)
        plt.xlabel("OverPressure (kPa)")
        plt.xlabel("Distance (m)")
        plt.grid()
        plt.title("B.S.T. Method")

        MLPlot = MCfigure.add_subplot(2,4,8)
        plt.ylabel("OverPressure (kPa)")
        plt.xlabel("Distance(m")
        plt.grid()
        plt.title("Machine Learning Method")

        #Finding maximum and minimum positions of distance curves for each method!
        CurveFit_max_index = np.where(CurveFitPressureArray == np.max(CurveFitPressureArray))
        TNT_max_index = np.where(TNTPressureArray == np.max(TNTPressureArray))
        TNO1_max_index = np.where(TNOPressureArray1 == np.max(TNOPressureArray1))
        TNO4_max_index = np.where(TNOPressureArray4 == np.max(TNOPressureArray4))
        TNO5_max_index = np.where(TNOPressureArray5 == np.max(TNOPressureArray5))
        TNO6_max_index = np.where(TNOPressureArray6 == np.max(TNOPressureArray6))
        BST_max_index = np.where(BSTPressureArray == np.max(BSTPressureArray))
        ML_max_index = np.where(MLPressureArray == np.max(MLPressureArray))

        CurveFit_min_index = np.where(CurveFitPressureArray == np.min(CurveFitPressureArray))
        TNT_min_index = np.where(TNTPressureArray == np.min(TNTPressureArray))
        TNO1_min_index = np.where(TNOPressureArray1 == np.min(TNOPressureArray1))
        TNO4_min_index = np.where(TNOPressureArray4 == np.min(TNOPressureArray4))
        TNO5_min_index = np.where(TNOPressureArray5 == np.min(TNOPressureArray5))
        TNO6_min_index = np.where(TNOPressureArray6 == np.min(TNOPressureArray6))
        BST_min_index = np.where(BSTPressureArray == np.min(BSTPressureArray))
        ML_min_index = np.where(MLPressureArray == np.min(MLPressureArray))

        CurveFitPressure = np.zeros((len(DistanceList)))    #Creating empty arrays to store the most probable/exact pressure curves
        TNTPressure = np.zeros((len(DistanceList)))
        TNOPressure1 = np.zeros((len(DistanceList)))
        TNOPressure4 = np.zeros((len(DistanceList)))
        TNOPressure5 = np.zeros((len(DistanceList)))
        TNOPressure6 = np.zeros((len(DistanceList)))
        BSTPressure = np.zeros((len(DistanceList)))
        MLPressure1 = np.zeros((len(DistanceList)))

        for distanceCounter in range(len(DistanceList)):    #calculating most probable result for each method

            CurveFitPressure[distanceCounter] = CurveFit(Fraction, Volume, DistanceList[distanceCounter])
            TNTPressure[distanceCounter] = TNT(Volume, DistanceList[distanceCounter])
            TNOPressure1[distanceCounter] = TNO(Volume, DistanceList[distanceCounter])[0] #TNO curve index (1,4,5,6) - [0,1,2,3]
            TNOPressure4[distanceCounter] = TNO(Volume, DistanceList[distanceCounter])[1]
            TNOPressure5[distanceCounter] = TNO(Volume, DistanceList[distanceCounter])[2]
            TNOPressure6[distanceCounter] = TNO(Volume, DistanceList[distanceCounter])[3]
            BSTPressure[distanceCounter] = BST(P1, Volume, DistanceList[distanceCounter])
            MLPressure1[distanceCounter] = MLpressure.predict([Fraction*100, Volume, DistanceList[distanceCounter]])[0]
        
        #PLotting most probable, max and min results of sensitivity analysis for each method!
        CurvefitPlot.plot(DistanceList,CurveFitPressure)
        CurvefitPlot.plot(DistanceList,CurveFitPressureArray[:, CurveFit_max_index[1][0]], linestyle="dashed")
        CurvefitPlot.plot(DistanceList,CurveFitPressureArray[:, CurveFit_min_index[1][0]], linestyle="dashed")
        CurvefitPlot.legend(["Most Probable","Maximum","Minimum"])

        TNTPlot.plot(DistanceList,TNTPressure)
        TNTPlot.plot(DistanceList,TNTPressureArray[:, TNT_max_index[1][0]], linestyle="dashed")
        TNTPlot.plot(DistanceList,TNTPressureArray[:, TNT_min_index[1][0]], linestyle="dashed")
        TNTPlot.legend(["Most Probable","Maximum","Minimum"])

        TNOPlot1.plot(DistanceList,TNOPressure1/1000)
        TNOPlot1.plot(DistanceList,TNOPressureArray1[:, TNO1_max_index[1][0]]/1000, linestyle="dashed")
        TNOPlot1.plot(DistanceList,TNOPressureArray1[:, TNO1_min_index[1][0]]/1000, linestyle="dashed")
        TNOPlot1.legend(["Most Probable","Maximum","Minimum"])

        TNOPlot4.plot(DistanceList,TNOPressure4/1000)
        TNOPlot4.plot(DistanceList,TNOPressureArray4[:, TNO4_max_index[1][0]]/1000, linestyle="dashed")
        TNOPlot4.plot(DistanceList,TNOPressureArray4[:, TNO4_min_index[1][0]]/1000, linestyle="dashed")
        TNOPlot4.legend(["Most Probable","Maximum","Minimum"])

        TNOPlot5.plot(DistanceList,TNOPressure5/1000)
        TNOPlot5.plot(DistanceList,TNOPressureArray5[:, TNO5_max_index[1][0]]/1000, linestyle="dashed")
        TNOPlot5.plot(DistanceList,TNOPressureArray5[:, TNO5_min_index[1][0]]/1000, linestyle="dashed")
        TNOPlot5.legend(["Most Probable","Maximum","Minimum"])

        TNOPlot6.plot(DistanceList,TNOPressure6/1000)
        TNOPlot6.plot(DistanceList,TNOPressureArray6[:, TNO6_max_index[1][0]]/1000, linestyle="dashed")
        TNOPlot6.plot(DistanceList,TNOPressureArray6[:, TNO6_min_index[1][0]]/1000, linestyle="dashed")
        TNOPlot6.legend(["Most Probable","Maximum","Minimum"])

        BSTPlot.plot(DistanceList,BSTPressure)
        BSTPlot.plot(DistanceList,BSTPressureArray[:, BST_max_index[1][0]], linestyle="dashed")
        BSTPlot.plot(DistanceList,BSTPressureArray[:, BST_min_index[1][0]], linestyle="dashed")
        BSTPlot.legend(["Most Probable","Maximum","Minimum"])

        MLPlot.plot(DistanceList, MLPressure1)
        MLPlot.plot(DistanceList,MLPressureArray[:, ML_max_index[1][0]], linestyle="dashed")
        MLPlot.plot(DistanceList,MLPressureArray[:, ML_min_index[1][0]], linestyle="dashed")
        MLPlot.legend(["Most Probable","Maximum","Minimum"])

    ImportantPlot() #Calling sub-function to create plot of most important analysis results

if MonteCarlo_On.casefold() == "yes":
    MonteCarlo()    #Calling Monte-Carlo simulation function

def FinalResults(): #Function for printing the distances to console
    
    for lim in P_limit:     #Printing results for  each pressure limit specified in the limit list
        #Finding the index location of the closest value of predicted pressure to the limit value
        print("For Pressure limit of: " + str(lim) + "kPa")
        curvefit_index = np.argmin(np.abs(np.subtract(CurveFitOverPressure,lim)))
        BST_index = np.argmin(np.abs(np.subtract(BSTOverPressure,lim)))
        ML_index = np.argmin(np.abs(np.subtract(MLOverPressure,lim)))
        #Finding the value of distance via linear interpolation between the closest values of pressure
        curvefit = np.interp(x=lim, xp=[CurveFitOverPressure[curvefit_index+1],CurveFitOverPressure[curvefit_index]],fp=[DistanceList[curvefit_index+1],DistanceList[curvefit_index]])
        BSTfit = np.interp(x=lim, xp=[BSTOverPressure[BST_index+1],BSTOverPressure[BST_index]],fp=[DistanceList[BST_index+1],DistanceList[BST_index]])
        MLfit = np.interp(x=lim, xp=[MLOverPressure[ML_index+1],MLOverPressure[ML_index]],fp=[DistanceList[ML_index+1],DistanceList[ML_index]])
        print("Curve Fitting: " + str(curvefit)[0:6] + "m")
        print("B.S.T. Method: " + str(BSTfit)[0:6] + "m")
        print("Machine Learn: " + str(MLfit)[0:6] + "m")
    
    print("------------------")

    for lim in H_limit:     #Printing results for each heat flux limit specified in the limit list
        print("For Heat Flux limit of: " + str(lim) + "kW/m2")
        solid_index = np.argmin(np.abs(np.subtract(SolidFlux,lim)))
        bleve_index = np.argmin(np.abs(np.subtract(BleveFlux,lim)))

        solid = np.interp(x=lim, xp=[SolidFlux[solid_index+1],SolidFlux[solid_index]],fp=[DistanceList[solid_index+1],DistanceList[solid_index]])
        bleve = np.interp(x=lim, xp=[BleveFlux[bleve_index+1],BleveFlux[bleve_index]],fp=[DistanceList[bleve_index+1],DistanceList[bleve_index]])
        print("Solid Flame: " + str(solid)[0:6] + "m")
        print("Bleve Flux: " + str(bleve)[0:6] + "m")

if Print_Limits_Console.casefold() == "yes":
    FinalResults()      #Calling final results printing function

#Timing END
print("Run Time: " + str(time.time() - startTime)[0:6] + "s")   #Ending timer and printing run-time (seconds)

plt.show()  #Showing plots