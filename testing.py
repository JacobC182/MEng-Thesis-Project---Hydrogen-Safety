import numpy as np

volume = 300
concentration = 0.3


#Distance range
distanceRange = range(5, 300 +1, 1)
distanceRange = np.array(distanceRange)

volumeList = []
concList = []

for i in range(len(distanceRange)):
    volumeList.append(volume)
    concList.append(concentration)

volumeList = np.array(volumeList)
concList = np.array(concList)

#np.insert(distanceRange, concentration, 1, axis=0)
ML_input = np.concatenate((concList, volumeList, distanceRange))

ML_input = np.reshape(ML_input, (3, len(distanceRange)))

#print(ML_input)

#print(np.linspace(10,1000,100,True))

#INPUT VALUES (FROM CFD)
Volume = 375    #cloud volume (m3)
Fraction = 0.05103  #Volume fraction of H2 (0>H2>1)

MaxSpreadCloud = 0.2     #Maximum +- % variability of Cloud Size (0<x<1)
MaxSpreadFraction = 0.2     #Maximum +- % variability of Volume Fraction (0<x<1)
n_Samples = 1000    #Number of random samples to use in simulation
#np.random.seed(0)
#VolumeDistribution = np.random.normal(loc=Volume, scale=MaxSpreadCloud*Volume, size=n_Samples)

#print(np.amin(VolumeDistribution))
#print(np.amax(VolumeDistribution))

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

#print(subsets)

subsets2.pop(0)
print(subsets2)
from itertools import permutations

subsets3 = []
for subset in subsets2:
  for x in permutations(subset):
    subsets3.append(x)

print(len(subsets3))
