import pandas as pd
import numpy as np
import seaborn as sns
import GMM
import EM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering, BisectingKMeans
from sklearn.metrics import homogeneity_score
from ucimlrepo import fetch_ucirepo 
from sklearn import preprocessing

def load_mines() -> pd.DataFrame:
    # fetch dataset 
    land_mines = fetch_ucirepo(id=763) 
    y = land_mines.data.targets
    X = land_mines.data.features
    # data (as pandas dataframes) 
    df = pd.concat([y.M, X], axis=1)
    return df


mines_em = load_mines()

mines_em['H'] += .0001
mines_em['S'] += .0001
X = np.array(mines_em[['H', 'V', 'S']])

gmm = GMM.GMM(10, 3)

# Initialize EM algo with data
gmm.init_em(X)
num_iters = 10
# Saving log-likelihood
log_likelihood = [gmm.log_likelihood(X)]
# plotting
for e in range(num_iters):
    # E-step
    gmm.e_step()
    # M-step
    gmm.m_step()
    # Computing log-likelihood
    log_likelihood.append(gmm.log_likelihood(X))
    print("Iteration: {}, log-likelihood: {:.4f}".format(e+1, log_likelihood[-1]))
    # plotting