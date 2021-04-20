import random
import numpy as np
import matplotlib.pyplot as plt

# opinion update function for agent i
def phi(x, a_0, a_1):
    return np.divide(x + a_0, np.array([1 for _ in range(N)]) + a_0 + a_1)

# initial utility for agent i
def u_0(x_0, a_0, a_1):
    x_0_plus = phi(x_0, a_0, a_1)
    return gamma[0] * np.matmul(rho, x_0_plus) - lambd[0] * np.matmul(np.array([1 for _ in range(N)]), a_0)

# initial values
gamma = [1, 1]
B = [10, 10]
lambd = [0.1, 0.1]
N = 100
b = [B[0] / 2, B[1] / 2]
x_0 = np.array([0.5 for index in range(N)])
pas = 0.01
a_init = np.array([[B[0] / N for _ in range(N)], [B[1] / N for _ in range(N)]])

# main loop
results = []
for C in [5, 10, 100]:
    result = []
    for number_of_leaders in range(10, 100, 5):
        leaders_indices = random.sample(range(N), number_of_leaders)
        rho = np.array([C if index in leaders_indices else 1 for index in range(N)])
        a = np.array([[B[0] / N for _ in range(N)], [B[1] / N for _ in range(N)]])
        
        # iterate long enough for convergence
        for x in range(100):
            W_0 = [
                [index for index in range(N) if a[0][index] == 0],
                [index for index in range(N) if a[1][index] == 0]
            ]
            W_1 = [
                [index for index in range(N) if a[0][index] == b[0]],
                [index for index in range(N) if a[1][index] == b[1]]
            ]

            # compute mu_0
            sqrt_d = [
                np.sqrt(rho * (np.array([1 for _ in range(N)]) - x_0 + a[1])),
                np.sqrt(rho * (x_0 + a[0]))
            ]
            sorting_condition = [
                np.divide(np.array([1 for _ in range(N)]) + a[1], sqrt_d[0]),
                np.divide(np.array([1 for _ in range(N)]) + a[0], sqrt_d[1])
            ]
            sorted_agents = [
                [t[0] for t in sorted(enumerate(sqrt_d[0]), key=lambda t: sorting_condition[0][t[0]])],
                [t[0] for t in sorted(enumerate(sqrt_d[1]), key=lambda t: sorting_condition[1][t[0]])],
            ]

            beta = [[0 for j in range(N)], [0 for j in range(N)]]
            for k in range(0, 2):
                # try to give water to everyone, remove people one by one
                # until we have enough water for everybody still here
                for i in range(N+1, 1, -1):
                    W_0 = sorted_agents[k][i:]  # no water to last [i, N]
                    W_1 = []  # no one saturated yet
                    W_2 = sorted_agents[k][:i]  # we fill all from 0 to i (for now)
                    saturated = True
                    mu_0 = 0

                    # try to put the same water level to everyone,
                    # find out who is saturated and adjust mu_0 iteratively
                    while saturated and len(W_2) > 1:
                        saturated = False

                        # compute current mu_0
                        numerator = sum([sqrt_d[k][index] for index in W_2])
                        denominator = B[k] - b[k] * len(W_1) + len(W_2) + sum([a[1-k][index] for index in W_2])
                        mu_0 = (numerator / denominator) ** 2 - lambd[k]

                        # update W_1 and W_2 based on saturations
                        new_W_1 = []
                        new_W_2 = []
                        for j in range(N):
                            if j in W_0:
                                beta[k][j] = 0
                            elif j in W_2:
                                beta[k][j] = np.sqrt(gamma[k] / (mu_0 + lambd[k])) * sqrt_d[k][j] - 1 - a[1-k][j]
                                if beta[k][j] > b[k]:
                                    saturated = True
                                    new_W_1.append(j)
                                else:
                                    new_W_2.append(j)
                            elif j in W_1:
                                beta[k][j] = b[k]
                                new_W_1.append(j)
                            else:
                                raise RuntimeError("should not happen")
                        W_1, W_2 = new_W_1, new_W_2

                    # if we have enough water for everyone, stop here
                    if sum(beta[0]) < B[0]:
                        break

            # gradient descent step
            a = [(beta[0] - a[0]) * pas + a[0], (beta[1] - a[1]) * pas + a[1]]

        # compute GoT
        got = (u_0(x_0, a[0], a_init[1]) - u_0(x_0, a_init[0], a_init[1])) / (u_0(x_0, a_init[0], a_init[1]))
        result.append(got)
    results.append(result)

# Initialise the figure and axes.
fig, ax = plt.subplots(1, figsize=(8, 6))

# Set the title for the figure
fig.suptitle('Gain by implementing the best response strategy', fontsize=15)

# Draw all the lines in the same plot, assigning a label for each one to be
# shown in the legend.
ax.plot(range(10, 100, 5), results[0], color="red", label="C=5")
ax.plot(range(10, 100, 5), results[1], color="green", label="C=10")
ax.plot(range(10, 100, 5), results[2], color="blue", label="C=100")

# Add a legend, and position it on the lower right (with no box)
plt.legend(loc="upper right", title="", frameon=False)

plt.show()
