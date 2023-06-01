import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# pathの設定
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
root_dir = os.path.dirname(current_dir)

input_dir = os.path.join(root_dir, 'input_data')
output_dir = os.path.join(root_dir, 'output_data')

csv_file = os.path.join(input_dir, 'dataset_weld_org.csv')

# CSVデータをロード
data = pd.read_csv(csv_file)

# x, y, zのデータを取得
x = data['welding_speed'].values
y = data['head_position'].values
z = data['penetration_depth'].values

# 最適化する関数を定義
def func(data, lambda_1, sigma, mu, A):
    x, y = data
    g_x = np.exp(lambda_1 * np.abs(x))
    f_y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((y - mu)**2) / (2 * sigma**2))
    h_xy = A * g_x * f_y
    return h_xy

# curve_fit関数を使用してパラメーターを最適化
popt, pcov = curve_fit(func, (x,y), z, p0=[0.0045, 2, 0, 1.0])

# 最適化されたパラメーターとそれらの誤差（共分散行列の対角成分の平方根）を出力
# print("Optimized parameters: ", popt)
# print("Parameter errors: ", np.sqrt(np.diag(pcov)))


# xとyの範囲と間隔を定義し、グリッドを作成
x_range = np.arange(20, 751, 10)
y_range = np.arange(-2.0, 2.1, 0.1)
X, Y = np.meshgrid(x_range, y_range)

# 最適化されたパラメータを使用してh_xyの値を計算
Z = func((X, Y), *popt)

# 2次元カラーマップを作成
fig, ax = plt.subplots()
c = ax.contourf(X, Y, Z, cmap='jet', vmin=0)

# グリッド、ラベル、カラーバーを設定
ax.grid(True)
ax.set_xlabel('welding speed (mm/sec)')
ax.set_ylabel('head position (mm)')
fig.colorbar(c, ax=ax, label='penetration depth (um)')

# グリッド、ラベル、カラーバーを設定
scatter = ax.scatter(x, y, c=z, cmap='jet', edgecolors='black')

# グラフの表示範囲を設定
ax.set_xlim([0, 770])
ax.set_ylim([-2.1, 2.1])

# プロットを表示します。
plt.show()

# CSVファイルとして出力します。
df = pd.DataFrame({
    'X': X.flatten(),
    'Y': Y.flatten(),
    'Z': Z.flatten()
})

csv_file = os.path.join(output_dir, 'output.csv')
df.to_csv('output.csv', index=False)