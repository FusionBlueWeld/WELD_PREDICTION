
import os
# GPUに関するエラーメッセージを非表示にする
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt



class WeldPredictor:
    def __init__(self, learning_rate, momentum):
        # 現在の作業ディレクトリを取得
        current_path = os.path.abspath(__file__)

        # ルートディレクトリを取得
        current_dir = os.path.dirname(current_path)
        root_dir = os.path.dirname(current_dir)

        # 各ディレクトリのパスを設定
        self.input_dir = os.path.join(root_dir, 'input_data')
        self.output_dir = os.path.join(root_dir, 'output_data')
        self.model_file_path = os.path.join(self.output_dir, 'weld_model.h5')

        # データセットのパスを設定
        self.dataset_path = os.path.join(self.input_dir, 'dataset_weld_optimized.csv')

        # 学習パラメータの設定
        self.learning_rate = learning_rate
        self.momentum = momentum

        # モデル初期化
        self.model = self.create_model()

    # データの読み込みと前処理
    def load_data(self):
        # データセットを読み込み
        df = pd.read_csv(self.dataset_path)

        # 説明変数と目的変数に分ける
        X = df[['welding_speed', 'head_position']].values
        y = df['penetration_depth'].values

        # 訓練データとテストデータに分ける（訓練データ：テストデータ = 8:2）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # データの標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def create_model(self):
        # モデルの定義
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=2, activation='relu'))    # 入力層
        self.model.add(Dropout(0.2))  # 過学習を防ぐためのドロップアウト層
        self.model.add(Dense(16, activation='relu'))                 # 隠れ層1
        self.model.add(Dropout(0.2))  # 過学習を防ぐためのドロップアウト層
        self.model.add(Dense(1, activation=None))               # 出力層

        # 最適化アルゴリズムとしてSGDを選択、学習率とモーメンタムはコンストラクタで設定された値を使用
        optimizer = SGD(learning_rate=self.learning_rate, momentum=self.momentum)

        # モデルのコンパイル
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', self.r2_score])
        return self.model

    # R^2スコアの定義
    def r2_score(self, y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))

    def calculate_dataset_stats(self):
        # データセットを読み込み
        df = pd.read_csv(self.dataset_path)

        # welding_speed と head_position の平均と標準偏差を計算
        self.mean_welding_speed = df['welding_speed'].mean()
        self.std_welding_speed = df['welding_speed'].std()
        self.mean_head_position = df['head_position'].mean()
        self.std_head_position = df['head_position'].std()

    def create_grid(self):
        # 軸の範囲と間隔
        x_range = np.arange(20, 750+1, 10)  # welding_speed range
        y_range = np.arange(-2.0, 2.0+0.1, 0.1)  # head_position range

        # グリッドを作成
        y_grid, x_grid = np.meshgrid(y_range, x_range)

        return x_grid, y_grid

    def standardize_grid(self, x_grid, y_grid):
        # 2Dグリッドを1D配列に変換
        x_flattened = x_grid.ravel()
        y_flattened = y_grid.ravel()

        # 標準化を実行
        x_flattened = (x_flattened - self.mean_welding_speed) / self.std_welding_speed
        y_flattened = (y_flattened - self.mean_head_position) / self.std_head_position

        return x_flattened, y_flattened

    def create_prediction_grid(self, x_flattened, y_flattened, x_grid_shape):
        # モデルの予測を実行
        z_flattened = self.model.predict(np.column_stack((x_flattened, y_flattened)))

        # 1D配列を2Dグリッドに戻す
        z_grid = z_flattened.reshape(x_grid_shape)

        return z_grid

    def plot_prediction_grid(self, x_grid, y_grid, z_grid):
        # カラーマップを作成
        plt.figure(figsize=(10, 7))
        plt.contourf(x_grid, y_grid, z_grid, cmap='jet')
        plt.colorbar(label='penetration depth (um)')
        plt.xlabel('welding speed (mm/sec)')
        plt.ylabel('head position (mm)')
        plt.grid(True)
        plt.show()

    def save_model(self):
        # 学習結果の保存
        self.model.save(self.model_file_path)

    def plot_r2_score(self, history):
        # R^2スコアの履歴を取得
        r2_score_history = history.history['val_r2_score']

        # R^2スコアのグラフ化
        plt.figure(figsize=(8, 6))
        plt.plot(r2_score_history)
        plt.title('R^2 Score by Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('R^2 Score')
        plt.legend(['Validation'])

        # グラフを画像ファイルとして保存
        plt.savefig(f'{self.output_dir}/r2_score.png')

    def train_model(self, X_train, y_train, X_test, y_test):
        # モデルの学習、verboseを0に設定して学習中の出力を抑制
        history = self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # データセットの統計情報を計算
        self.calculate_dataset_stats()

        # グリッドを作成
        x_grid, y_grid = self.create_grid()

        # グリッドを標準化
        x_flattened, y_flattened = self.standardize_grid(x_grid, y_grid)

        # 予測値のグリッドを作成
        z_grid = self.create_prediction_grid(x_flattened, y_flattened, x_grid.shape)

        # 予測値のグリッドをプロット
        self.plot_prediction_grid(x_grid, y_grid, z_grid)

        # 学習済みモデルを保存
        self.save_model()

        # R^2スコアをプロット
        self.plot_r2_score(history)


class WeldFineTuner(WeldPredictor):
    def __init__(self, learning_rate, momentum):
        super().__init__(learning_rate, momentum)
        # 新たなデータセットのパスを設定
        self.dataset_path = os.path.join(self.input_dir, 'dataset_weld_org.csv')

    # R^2スコアの定義
    def r2_score(self, y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))

    def load_trained_model(self):
        # 学習済みモデルを読み込む
        self.model = load_model(self.model_file_path, custom_objects={'r2_score': self.r2_score})


    def fine_tune(self):
        # データをロードして学習データとテストデータに分割
        X_train, X_test, y_train, y_test = self.load_data()

        # モデルを訓練し、結果を保存
        self.train_model(X_train, y_train, X_test, y_test)

        # ファインチューニング後のモデルを保存
        self.model_file_path = os.path.join(self.output_dir, f'weld_model_finetuned.h5')
        self.model.save(self.model_file_path)

        # R^2スコアのグラフを保存
        plt.savefig(f'{self.output_dir}/r2_score_finetuned.png')

        # 2次元カラーマップのグラフを保存
        plt.savefig(f'{self.output_dir}/2D_colormap_finetuned.png')

def main():
    # WeldPredictor クラスのインスタンスを作成
    weld_predictor = WeldPredictor(0.00005, 0.9)

    # データをロードして学習データとテストデータに分割
    X_train, X_test, y_train, y_test = weld_predictor.load_data()

    # モデルを訓練し、結果を保存
    weld_predictor.train_model(X_train, y_train, X_test, y_test)

    # WeldFineTuner クラスのインスタンスを作成
    weld_fine_tuner = WeldFineTuner(0.00005, 0.9)

    # 学習済みモデルをロード
    weld_fine_tuner.load_trained_model()

    # 追加学習を実行
    weld_fine_tuner.fine_tune()

if __name__ == "__main__":
    main()
