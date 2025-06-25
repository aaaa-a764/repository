import numpy as np
import numpy.linalg as LA


def PCA(x, y):
    N = x.shape[0]  # データの行数
    d = x.shape[1]  # 次元数
    μ = np.zeros(d)   # 平均ベクトル
    s = np.zeros((d, d))   # 分散共分散行列
    z = np.zeros(d)    # 主成分
    for m in range(d):
        for i in range(N):
            μ[m] += x[i][m] / N
    for m in range(d):
        for k in range(d):
            for i in range(N):
                s[k][m] += (x[i][k] - μ[k]) * (x[i][m] - μ[m]) / N
    λ, v = LA.eig(s)
    sorted_indices = np.argsort(λ)[::-1]  # 固有値を降順にソートするインデックス
    λ = λ[sorted_indices]
    v = v[:, sorted_indices]  # 固有ベクトルも対応して並べ替える

    # 主成分スコアの計算
    z = np.dot(x - μ, v)  # データを固有ベクトルに投影

    kiyo = np.zeros(d)
    kiyo[y] = λ[y] / sum(λ)

    ruiseki = np.zeros(d)
    ruiseki[y] = sum(λ[0: y + 1]) / sum(λ)

    return z[:, y], kiyo[y], ruiseki[y]


deta = np.array([[6, 9, 4], [2, 5, 7], [8, 5, 6], [3, 5, 4], [7, 4, 9],
                 [4, 3, 4], [3, 6, 8], [6, 8, 2], [5, 4, 5], [6, 7, 6]])

result1 = PCA(deta, 0)
result2 = PCA(deta, 1)
result3 = PCA(deta, 2)

print("第1主成分:" + str(result1[0]))
print("式：0.5986(x1-5.2)+0.5843(x2-5.6)+0.548(x3-5.5)")
print("寄与率:" + str(result1[1]))
print("累積寄与率" + str(result1[2]))
print()

print("第2主成分:" + str(result2[0]))
print("式：0.683(x1-5.2)-0.0148(x2-5.6)+0.7303(x3-5.5)")
print("寄与率:" + str(result2[1]))
print("累積寄与率" + str(result2[2]))
print()

print("第3主成分:" + str(result3[0]))
print("式：-0.4185(x1-5.2)+0.8114(x2-5.6)+0.4079(x3-5.5)")
print("寄与率:" + str(result3[1]))
print("累積寄与率" + str(result3[2]))
print()
