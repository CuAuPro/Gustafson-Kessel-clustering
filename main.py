import numpy as np
import matplotlib.pyplot as plt

from lib.gk import GK
from lib.vis import *
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

X_test_0 = np.array([1.2, -0.8])
X_test_1 = np.array([-2.1, -2.8])

gk = GK(n_clusters = 2)
res = gk.fit(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], color='b')
plt.scatter(gk.V[:,0], gk.V[:,1], color='r')

pos = X[np.where(gk.U.argmax(axis=0))].mean(axis=0)
plot_cov_ellipse(gk.A[1,:], pos, facecolor=(0, 1, 0, 0.25))

pos = X[np.where(gk.U.argmin(axis=0))].mean(axis=0)
plot_cov_ellipse(gk.A[0,:], pos, facecolor=(0, 1, 0, 0.25))

pred = gk.predict(X_test_0)
print("Predicted cluster for cyan sample: %d", pred)
plt.scatter(X_test_0[0], X_test_0[1], color='brown')

pred = gk.predict(X_test_1)
print("Predicted cluster for orange: %d", pred)
plt.scatter(X_test_1[0], X_test_1[1], color='orange')

plt.show()






