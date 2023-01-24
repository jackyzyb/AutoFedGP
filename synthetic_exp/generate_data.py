import numpy as np

# settings
n = 50  # dim of x
d = 10  # dim of y
m = 100     # num of feature vectors
A = np.random.random([d, m]) - 0.5
b = np.random.random([d, 1]) - 0.5
N = 5000   # num of samples
num_centers = 10

# generate x
# a mixture of num_centers Gaussian
centers = np.random.random([n, num_centers]) - 0.5
X = np.random.normal(0, 1, size=(n, N))
for center in range(num_centers):
    # deviate the samples
    stride = int(N / num_centers)
    X[:, center * stride: (center + 1) * stride] += centers[:, center].reshape([-1, 1])

# normalize the data to be within the unit ball
# r_max = (sum(X * X) ** 0.5).max()
# X /= r_max

# generate rbf feature functions
mu = np.random.random([n, m]) - 0.5
sigma_square = 100 * np.random.random(m)

for source_target_diff in list(np.arange(0., 1., 0.07)):

    mu_target = mu + np.random.random([n, m]) * source_target_diff
    sigma_square_target = sigma_square + 100 * np.random.random(m) * source_target_diff

    # generate features
    features = np.zeros([m, N])
    for i in range(m):
        t = (mu_target[:, i].reshape([-1, 1]) - X)
        temp = sum(t ** 2)
        features[i, :] = np.exp((- sum(t ** 2)) / sigma_square_target[i])

    # generate Y
    Y = A.dot(features) + b

    # save_name = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(num_centers)
    # np.savez('datasets/' + save_name, X=X, Y=Y)

    save_name = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(
        num_centers) + 'source_target_diff-' + str(source_target_diff)
    np.savez('datasets/' + save_name, X=X, Y=Y)

    # # slighted changed
    # save_name = 'alter-' + 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(num_centers)
    # shift = np.random.random([X.shape[0], 1]) - 0.5
    # np.savez('datasets/' + save_name, X=X + shift, Y=Y)
