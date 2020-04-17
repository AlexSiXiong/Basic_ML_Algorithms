import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        assert x.ndim == 2, '只处理二维数组'

        self.mean_ = np.array([np.mean(x[:, i]) for i in range(x.shape[1])]) # 下面有具体讲解的
        self.scale_ = np.array([np.std(x[:, i]) for i in range(x.shape[1])])

        return self

    def transform(self, x):
        assert x.ndim == 2
        assert self.scale_ is not None and self.mean_ is not None
        assert x.shape[1] == len(self.mean_)

        result = np.empty(shape=x.shape, dtype=float)
        for col in range(x.shape[1]):
            result[:, col] = (x[:, col] - self.mean_[col]) / self.scale_
        return result


"""
讲解上面那个怎么计算的
"""
x = np.random.randint(0, 10, size=(7,2))

print(x.shape[0])  # row 数量
print(x.shape[1])  # col 数量

# print(x)

# print(np.mean(x[:, 1]))  # 这个是求col = 1 的平均值

# print(np.array([np.mean(x[:, i]) for i in range(x.shape[1])])) # 这个是求col = 1和2的平均值