import torch
import matplotlib.pyplot as plt

def proj(v, u, counter):
    # project u to the direction of v
    a = torch.sum(v * u) / torch.sum(v * v)
    if a < 0 :
        a = 0
        counter[0] += 1
    return  a * v

def proj_batch(V, u, proj_direction):
    # project u to many vectors in V (shape (N, d))
    N, d = V.shape
    V_gp = torch.zeros(N, d)
    counter = [0]
    for n in range(N):
        if proj_direction == 'proj_source_to_target':
            V_gp[n, :] = proj(V[n, :], u, counter)
        elif proj_direction == 'proj_target_to_source':
            V_gp[n, :] = proj(u, V[n, :], counter)
        else:
            raise NameError
    print("number of cases where projection is projection is negative: ", counter[0])
    return V_gp

def expected_squared_dist(mu, samples):
    # samples is of shape (N, d)
    N = samples.shape[0]
    return (torch.sum((samples - mu) ** 2) / N).item()

def cosine_dist(mu, samples):
    N = samples.shape[0]
    cos = torch.sum(mu * samples, dim=1) / torch.norm(mu) / torch.norm(samples, dim=1)
    return (torch.sum(cos) / N).item()

#####################
#### Settings #######
#####################
d = 1000
N = 500
std_T = 0.5
diff_S_T = 1.5
mu_T = torch.normal(0, torch.ones(d))
samples_T = torch.normal(mu_T.repeat(N, 1), std_T * torch.ones(N, d))
mu_S = torch.normal(mu_T, diff_S_T * torch.ones(d))
# mu_S = mu_S / torch.norm(mu_S) * torch.norm(mu_T)

##############################################
#### compute expected squared distance #######
##############################################
beta_list = [0.1 * i for i in range(11)]
expected_squared_dist_convex_list = []
expected_squared_dist_gp_S_to_T_list = []
expected_squared_dist_gp_T_to_S_list = []
# cosine_dist_convex_list = []
# cosine_dist_gp_list = []
projections_S_to_T = proj_batch(samples_T, mu_S, 'proj_source_to_target')
projections_T_to_S = proj_batch(samples_T, mu_S, 'proj_target_to_source')
for beta in beta_list:
    g_convex = beta * mu_S + (1 - beta) * samples_T
    g_gp_S_to_T = beta * projections_S_to_T + (1 - beta) * samples_T
    g_gp_T_to_S = beta * projections_T_to_S + (1 - beta) * samples_T
    expected_squared_dist_convex_list.append(expected_squared_dist(mu_T, g_convex))
    expected_squared_dist_gp_S_to_T_list.append(expected_squared_dist(mu_T, g_gp_S_to_T))
    expected_squared_dist_gp_T_to_S_list.append(expected_squared_dist(mu_T, g_gp_T_to_S))
    #cosine_dist_convex_list.append(cosine_dist(mu_T, g_convex))
    # cosine_dist_gp_list.append(cosine_dist(mu_T, g_gp))

print(expected_squared_dist_convex_list)
print(expected_squared_dist_gp_S_to_T_list)
print(expected_squared_dist_gp_T_to_S_list)
# print(cosine_dist_convex_list)
# print(cosine_dist_gp_list)


#####################
#### Plot #######
#####################

plt.plot(beta_list, expected_squared_dist_convex_list, label='convex comb')
plt.plot(beta_list, expected_squared_dist_gp_S_to_T_list, label='gp proj S to T')
plt.plot(beta_list, expected_squared_dist_gp_T_to_S_list, label='gp proj T to S')
plt.legend(loc="upper center")
plt.title('Results (std_T={}; diff_S_T={})'.format(std_T, diff_S_T))
plt.ylabel('expected squared distance')
plt.xlabel('beta')
plt.show()