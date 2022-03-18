import numpy as np
import openpyxl
import matplotlib.pyplot as plt

w1 = openpyxl.load_workbook("CV-Results.xlsx")
s = w1.active

for column in s.iter_cols(min_row=3, max_row=2162, min_col=13,max_col=13, values_only=True):
    GB_data = column

index = np.where(GB_data == np.max(GB_data))[0][0]

plt.figure(1)
plt.scatter(index+1,GB_data[index], s=200, marker="*")
plt.scatter([*range(1,len(GB_data)+1, 1)],GB_data, s=10)
plt.xlabel("Model Iteration")
plt.ylabel("R2 Score")
plt.title("Model Training Scores")
plt.legend(['Best Model'])
plt.grid()
plt.show()

