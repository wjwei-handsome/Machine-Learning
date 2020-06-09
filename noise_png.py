from sklearn import datasets
import matplotlib.pyplot as plt 
import os

dirname='noise_test_pngs/'
if dirname  not in os.listdir('.'):
    os.mkdir(dirname)
for noise in range(1,101,5):
    wwj_X,wwj_y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=noise)
    plt.scatter(wwj_X,wwj_y)
    plt.savefig(dirname+'noise'+str(noise)+'.png')
    plt.close('all')