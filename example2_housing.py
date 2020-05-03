'''
import os
import pandas as pd
import numpy as np 
##加载数据
housing=pd.read_csv('datasets/housing/housing.csv')
housing.info()
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot
plt.savefig('test.png')