import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import numpy as np

beta_list = list(np.arange(0, 1, 1/100))
sigma_to_d_list = list(np.arange(0, 1.3, 1.3/100))

# print(len(sigma))
convex_beta = []
convex_d = []
gp_beta = []
gp_d = []
target_beta = []
target_d = []
source_beta = []
source_d = []

methods = [[convex_beta, convex_d], [gp_beta, gp_d], [target_beta, target_d], [source_beta, source_d]]


cos_theta = 0.7

for beta in beta_list:
    for sigma_to_d in sigma_to_d_list:
        convex = beta ** 2  + (1-beta) ** 2 * sigma_to_d ** 2
        gp = beta ** 2 * cos_theta ** 2  + ((1-beta)**2 + (2*beta-beta**2)/10) * sigma_to_d ** 2
        tg =    sigma_to_d ** 2
        src = 1
        idx = np.argsort([convex, gp, tg, src])
        choice = int(idx[0])
        # if tg > convex:
        methods[choice][0].append(beta)
        methods[choice][1].append(sigma_to_d)

plt.plot(gp_d, gp_beta, 'bo', label='Gradient Projection')
plt.plot(convex_d, convex_beta, 'ro', label='Convex Combination')
plt.plot(target_d, target_beta, 'go', label='Target Only')
plt.plot(source_d, source_beta, 'yo', label='Source Only')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Smallest Upper Bound')
plt.xlabel('Relative Target Domain Variance')
plt.ylabel('beta')
plt.show()