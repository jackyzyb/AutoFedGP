import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import numpy as np

sigma = list(np.arange(0, 1, 0.01))
ds = list(np.arange(0, 1, 0.01))

# print(len(sigma))
convex_sigma = []
convex_d = []
gp_sigma = []
gp_d = []
target_sigma = []
target_d = []
source_sigma = []
source_d = []
fly_sigma = []
fly_d = []

methods = [[convex_sigma, convex_d], [gp_sigma, gp_d], [target_sigma, target_d], [source_sigma, source_d], [fly_sigma, fly_d]]


cos_theta = 0.7

for s in sigma:
    for d in ds:
        convex = 0.25 * d ** 2 + 0.25 * s ** 2
        gp = (d ** 2 * 0.25 ) * cos_theta ** 2  + (0.25+0.75/10) * s ** 2
        tg =  s ** 2
        src = d ** 2
        # fly = (0.5 * (s  + d )) ** 2
        idx = np.argsort([convex, gp, tg, src])
        choice = int(idx[0])
        # if tg > convex:
        methods[choice][0].append(s)
        methods[choice][1].append(d)

plt.plot(gp_d, gp_sigma, 'bo', label='Gradient Projection')
plt.plot(convex_d, convex_sigma, 'ro', label='Convex Combination')
plt.plot(target_d, target_sigma, 'go', label='Target Only')
plt.plot(source_d, source_sigma, 'yo', label='Source Only')
plt.plot(fly_d, fly_sigma, 'mo', label='On The Fly')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Smallest Upper Bound')
plt.ylabel('Target Domain Variance')
plt.xlabel('Source-Target Domain Difference')
plt.show()