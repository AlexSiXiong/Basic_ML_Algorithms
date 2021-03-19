import matplotlib.pyplot as plt
import numpy as np

#
# plot_x = np.linspace(-1, 6, 141)
# plot_y = (plot_x - 2.5) ** 2 - 1
#
# plt.plot(plot_x, plot_y)
# plt.show()
#
# lr = 0.1
# epsilon = 1e-8
#
#
# def dJ(theta):
#     return 2 * (theta - 2.5)
#
#
# def J(theta):
#     return (theta - 2.5) ** 2 - 1
#
#
# theta = 0.0

# while True:
#     gradient = dJ(theta)
#     last_theta = theta
#     theta = theta - gradient * lr
#     if abs(J(theta) - J(last_theta)) < epsilon:
#         break
#
# print(theta)
# print(J(theta))
np.random.seed(3)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)

X = x.reshape(-1, 1)
plt.scatter(x, y)
plt.show()


def J(theta, X_b, y):
    # loss func
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except:
        return float('inf')


def dJ(theta, X_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
    return res * 2 / len(X_b)


def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = dJ(theta, X_b, y)

        last_theta = theta
        theta = theta - eta * gradient
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break

        cur_iter += 1

    return theta


X_b = np.hstack([np.ones((len(x), 1)), x.reshape(-1, 1)])
print('haha')
print(X_b.shape)
initial_theta = np.zeros(X_b.shape[1])
print(initial_theta.shape)
eta = 0.01

theta = gradient_descent(X_b, y, initial_theta, eta)

