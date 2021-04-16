import random
import numpy as np
import matplotlib.pyplot as plt


# def sqrt_d_i(i, a):
#     if i==0:
#         return np.sqrt(rho*(np.array([1 for _ in range(N)])-x_0+ a[1]))
#     if i==1:
#         return np.sqrt(rho*(x_0+ a[0]))

# def mu_0_i(i, a, W_1, W_2, sqrt_d):
#     if i==0:
#         return (sum([sqrt_d[0][index] for index in W_2[0]])/ (B[0]-b[0]*len(W_1[0])+len(W_2[0])+sum([a[1][index] for index in W_2[0]])))**2 - lambd[0]
#     if i==1:
#         return (sum([sqrt_d[1][index] for index in W_2[1]])/ (B[1]-b[1]*len(W_1[1])+len(W_2[1])+sum([a[0][index] for index in W_2[1]])))**2 - lambd[1]

# def alpha_i(i, mu_0, sqrt_d, a):
#     if i==0:
#         return sqrt_d[0]*np.sqrt(gamma[0]/(mu_0[0]+lambd[0]))  - a[1]
#     if i==1:
#         return sqrt_d[1]*np.sqrt(gamma[1]/(mu_0[1]+lambd[1]))  - a[0]

# def beta_i(i, a, alpha):
#     if i==0:
#         return np.minimum([b[0] for index in range(N)], np.maximum([0 for index in range(N)], alpha[0]))
#     if i==1:
#         return np.minimum([b[1] for index in range(N)], np.maximum([0 for index in range(N)], alpha[1]))

def phi(x, a_0, a_1):
    return np.divide(x + a_0, np.array([1 for _ in range(N)]) + a_0 + a_1)

def u_0(x_0, a_0, a_1):
    gamma_0 = gamma[0]
    lambd_0 = lambd[0]
    x_0_plus = phi(x_0, a_0, a_1)
    return gamma_0 * np.matmul(rho, x_0_plus) - lambd_0 * np.matmul(np.array([1 for _ in range(N)]), a_0)


gamma = [1, 1]
B = [10, 10]
lambd = [0.1, 0.1]
N = 100
b = [B[0] / 10, B[1] / 10]
x_0 = np.array([0.5 for index in range(N)])
pas = 0.01
a_init = np.array([[B[0] / N for _ in range(N)], [B[1] / N for _ in range(N)]])

results = []
for C in [5, 10, 100]:
    # C=5
    result = []
    for number_of_leaders in range(1, 100, 5):
        # number_of_leaders=13
        leaders_indices = random.sample(range(N), number_of_leaders)
        print(leaders_indices)
        rho = np.array([C if index in leaders_indices else 1 for index in range(N)])
        a = np.array([[B[0] / N for _ in range(N)], [B[1] / N for _ in range(N)]])
        # print(a[0])
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
            # print(sorting_condition)
            sorted_agents = [
                [t[0] for t in sorted(enumerate(sqrt_d[0]), key=lambda t: sorting_condition[0][t[0]])],
                [t[0] for t in sorted(enumerate(sqrt_d[1]), key=lambda t: sorting_condition[1][t[0]])],
            ]
            #print(sorted_agents)

            beta = [[0 for j in range(N)], [0 for j in range(N)]]
            for i in range(N+1, 1, -1):
                W_0 = sorted_agents[0][i:]  # no water to last [i, N]
                W_1 = []  # no one saturated yet
                W_2 = sorted_agents[0][:i]  # we fill all from 0 to i for now
                saturated = True
                mu_0 = 0

                while saturated and len(W_2) > 1:
                    saturated = False

                    # compute current mu_0
                    numerator = sum([sqrt_d[0][index] for index in W_2])
                    denominator = B[0] - b[0] * len(W_1) + len(W_2) + sum([a[1][index] for index in W_2])
                    mu_0 = (numerator / denominator) ** 2 - lambd[0]

                    # update W_1 and W_2 based on saturations
                    new_W_1 = []
                    new_W_2 = []
                    for j in range(N):
                        if j in W_0:
                            beta[0][j] = 0
                        elif j in W_2:
                            beta[0][j] = np.sqrt(gamma[0] / (mu_0 + lambd[0])) * sqrt_d[0][j] - 1 - a[1][j]
                            if beta[0][j] > b[0]:
                                saturated = True
                                new_W_1.append(j)
                            else:
                                new_W_2.append(j)
                        elif j in W_1:
                            beta[0][j] = b[0]
                            new_W_1.append(j)
                        else:
                            raise RuntimeError("bite")
                    W_1, W_2 = new_W_1, new_W_2

                if sum(beta[0]) < B[0]:
                    break

            for i in range(N + 1, 1, -1):
                W_0 = sorted_agents[1][i:]  # no water to last [i, N]
                W_1 = []  # no one saturated yet
                W_2 = sorted_agents[1][:i]  # we fill all from 0 to i for now
                saturated = True
                mu_0 = 0

                while saturated and len(W_2) > 1:
                    saturated = False

                    # compute current mu_0
                    numerator = sum([sqrt_d[1][index] for index in W_2])
                    denominator = B[1] - b[1] * len(W_1) + len(W_2) + sum([a[0][index] for index in W_2])
                    mu_0 = (numerator / denominator) ** 2 - lambd[1]

                    # update W_1 and W_2 based on saturations
                    new_W_1 = []
                    new_W_2 = []
                    for j in range(N):
                        if j in W_0:
                            beta[1][j] = 0
                        elif j in W_2:
                            beta[1][j] = np.sqrt(gamma[1] / (mu_0 + lambd[1])) * sqrt_d[1][j] - 1 - a[0][j]
                            if beta[1][j] > b[1]:
                                saturated = True
                                new_W_1.append(j)
                            else:
                                new_W_2.append(j)
                        elif j in W_1:
                            beta[1][j] = b[1]
                            new_W_1.append(j)
                        else:
                            raise RuntimeError("bite")
                    W_1, W_2 = new_W_1, new_W_2

                if sum(beta[1]) < B[1]:
                    break

                # numerator = sum([sqrt_d[0][index] for index in W_2])
                # denominator = B[0] - b[0] * len(W_1[0]) + len(W_2) + sum([a[1][index] for index in W_2])
                # mu_0 = (numerator / denominator) ** 2 - lambd[0]
                # beta = []
                # for n in range(N):
                #     if n in W_0[0]:
                #         beta.append(0)
                #     elif n in W_1[0]:
                #         beta.append(b[0])
                #     else:
                #         beta.append(np.sqrt(gamma[0] / (mu_0 + lambd[0])) * sqrt_d[0][n] - 1 - a[1][n])
                # if sum(beta) >= B[0]:
                #     break
                # a_next[0] = (beta - a[0]) * pas + a[0]

            # for i in range(1, N+1):
            #     W_2 = sorted_agents[1][:i]
            #     numerator = sum([sqrt_d[1][index] for index in W_2])
            #     denominator = B[1] - b[1] * len(W_1[0]) + len(W_2) + sum([a[0][index] for index in W_2])
            #     mu_0 = (numerator / denominator) ** 2 - lambd[1]
            #     alpha = np.sqrt(gamma[1] / (mu_0 + lambd[1])) * sqrt_d[1] - 1 - a[0]
            #     beta = np.minimum([b[1] for index in range(N)], np.maximum([0 for index in range(N)], alpha[1]))
            #     if sum(beta) >= B[1]:
            #         break
            #     a_next[1] = (beta - a[1]) * pas + a[1]

            a = [(beta[0] - a[0]) * pas + a[0], (beta[1] - a[1]) * pas + a[1]]

            """
            numerator = [
                sum([sqrt_d[0][index] for index in W_2[0]]),
                sum([sqrt_d[1][index] for index in W_2[1]])]
            denominator = [
                B[0] - b[0] * len(W_1[0]) + len(W_2[0]) + sum([a[1][index] for index in W_2[0]]),
                B[1] - b[1] * len(W_1[1]) + len(W_2[1]) + sum([a[0][index] for index in W_2[1]])
            ]

            mu_0 = [
                (numerator[0] / denominator[0]) ** 2 - lambd[0],
                (numerator[1] / denominator[1]) ** 2 - lambd[1]
            ]
            # print(mu_0)
            # if number_of_leaders <= 15:
            #     print(leaders_indices, mu_0)
            # compute alpha

            alpha = [
                np.sqrt(gamma[0] / (mu_0[0] + lambd[0])) * sqrt_d[0] - 1 - a[1],
                np.sqrt(gamma[1] / (mu_0[1] + lambd[1])) * sqrt_d[1] - 1 - a[0]
            ]

            beta = [
                np.minimum([b[0] for index in range(N)], np.maximum([0 for index in range(N)], alpha[0])),
                np.minimum([b[1] for index in range(N)], np.maximum([0 for index in range(N)], alpha[1]))
            ]
            a = [(beta[0] - a[0]) * pas + a[0], (beta[1] - a[1]) * pas + a[1]]
            # if x in [0, 9] and number_of_leaders<=20:
            # #print(leaders_indices, sqrt_d, mu_0, alpha, beta, a, sep='\n')
            #     print("SUMA: ", x,  sum(a[0]), '\n\n')
            #     if sum(a[0]) > 10:
            #         print(a[0])
            """

        # compute GoT
        # print('here', beta[0], a[0])
        got = (u_0(x_0, a[0], a_init[1]) - u_0(x_0, a_init[0], a_init[1])) / (u_0(x_0, a_init[0], a_init[1]))
        result.append(got)
    results.append(result)

# Initialise the figure and axes.
fig, ax = plt.subplots(1, figsize=(8, 6))

# Set the title for the figure
fig.suptitle('Multiple Lines in Same Plot', fontsize=15)

# Draw all the lines in the same plot, assigning a label for each one to be
# shown in the legend.
ax.plot(range(1, 100, 5), results[0], color="red", label="C=5")
ax.plot(range(1, 100, 5), results[1], color="green", label="C=10")
ax.plot(range(1, 100, 5), results[2], color="blue", label="C=100")

# Add a legend, and position it on the lower right (with no box)
plt.legend(loc="upper right", title="", frameon=False)

plt.show()
