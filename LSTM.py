import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl
import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.stats as stats
import statsmodels.api as sm
import tqdm
import os
import glob
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
    LeaveOneOut
)

import torch
import torch.nn as nn
import torch.optim as optim


import optuna
from sklearn.pipeline import Pipeline

from scipy.stats import f_oneway, tukey_hsd, ttest_ind,mannwhitneyu
from scipy import stats
from sklearn.model_selection import GroupKFold
import time
from collections import Counter

from scipy import signal

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Masking
from keras.layers import LSTM, Bidirectional
import tensorflow as tf
from collections import Counter

def convert_vec(data_x,data_y,data_z):
    return np.sqrt(
        data_x ** 2 + data_y ** 2 + data_z ** 2)

def diffentiation_signal(data):
    dy = np.gradient(data)
    return dy

def fourier_tarnsform(data,dt=1.0/200):
    F = np.fft.fft(data)
    N = len(data)
    # 周波数軸
    freq = np.fft.fftfreq(N, d=dt)
 
    # 振幅スペクトル
    Amp = np.abs(F / (N / 2))
    return freq,Amp

home_path="../../Data/OpenData/new_csv_Data"
dirs = os.listdir(home_path)
target_list=list()
dfs=list()
for d in dirs:
    dir_path = f"{home_path}/{d}/"
    file_list = os.listdir(dir_path)
    for f in file_list:
        if ".csv" in f:
            target_list.append(f)
            path=os.path.join(dir_path,f)
            df = pd.read_csv(path)
            print(path)
            dfs.append(df)

new_dfs = []
new_columns = [
        "Thumb_x_vel","Thumb_y_vel","Thumb_z_vel","Index_x_vel","Index_y_vel","Index_z_vel",
            "Thumb_vec_vel","Index_vec_vel","Thumb2Index_vec_vel",
            "Thumb_x_acc","Thumb_y_acc","Thumb_z_acc",
            "Thumb_vec_acc",
            "Index_x_acc","Index_y_acc","Index_z_acc",
            "Index_vec_acc",
            "Thumb2Index_vec_acc",
               "Timestamp"
              ]

remove_dfs = []
new_target_list = []
out_target_list = []
max_len_sequence = 0
X =[]
for i,df in enumerate(dfs):
    new_signal = []

    Thumb_x = df["Thumb_X"].values
    Thumb_y = df["Thumb_Y"].values
    Thumb_z = df["Thumb_Z"].values
    Index_x = df["Index_X"].values
    Index_y = df["Index_Y"].values
    Index_z = df["Index_Z"].values
    if len(set(Thumb_x)) == 1 or len(set(Thumb_y)) == 1 or len(set(Thumb_z)) == 1 or len(set(Index_x)) == 1 or len(set(Index_y)) == 1 or len(set(Index_z)) == 1:
        remove_dfs.append(df)
        out_target_list.append(target_list[i])
        continue
    new_signal.extend([Thumb_x,Thumb_y,Thumb_z,Index_x,Index_y,Index_z])

    # vector
    Thumb_vec = convert_vec(Thumb_x,Thumb_y,Thumb_z)
    Index_vec = convert_vec(Index_x,Index_y,Index_z)
    Thumb2Index_vec = convert_vec(Thumb_x-Index_x,Thumb_y-Index_y,Thumb_z-Index_z) # our propose
    
    new_signal.extend([Thumb_vec,Index_vec,Thumb2Index_vec])

    # Angular acceleration
    Thumb_x_acc = diffentiation_signal(Thumb_x)
    Thumb_y_acc = diffentiation_signal(Thumb_y)
    Thumb_z_acc = diffentiation_signal(Thumb_z)
    Thumb_vec_acc = diffentiation_signal(Thumb_vec)

    Index_x_acc = diffentiation_signal(Index_x)
    Index_y_acc = diffentiation_signal(Index_y)
    Index_z_acc = diffentiation_signal(Index_z)
    Index_vec_acc = diffentiation_signal(Index_vec)

    Thumb2Index_vec_acc = diffentiation_signal(Thumb2Index_vec)

    new_signal.extend([Thumb_x_acc,Thumb_y_acc,Thumb_z_acc,
                       Thumb_vec_acc,
                       Index_x_acc,Index_y_acc,Index_z_acc,
                       Index_vec_acc,
                   Thumb2Index_vec_acc])
    
    timestamp = np.array(list(range(len(Thumb_x)))) * 0.005 #200Hz
    new_signal.extend([timestamp])
    # print(target_list[i])
    if max_len_sequence < len(timestamp):
        max_len_sequence = len(timestamp)
        
    new_signal = np.array(new_signal,dtype = 'float')
    X.append(new_signal[:-1].T)
    new_df = pd.DataFrame(new_signal.T,columns=new_columns)
    
    new_dfs.append(new_df)
    new_target_list.append(target_list[i])

from keras.preprocessing.sequence import pad_sequences
# パディングを適用して nd.array に変換
padding_value = -9999
X_padded = pad_sequences(X, maxlen=max_len_sequence, dtype='float32', padding='post', truncating='post', value=padding_value)

y = []
subject_list = []
for target in new_target_list:
    disease,subject_id,_ = target.split('_')
    if disease == "CTRL":
        label = 0
    elif disease == "PD":
        label = 1
    elif disease == "PSP":
        label = 2
    elif disease == "MSA":
        label = 3
    y.append(label)
    subject_info = f"{disease}_{subject_id}"
    subject_list.append(subject_info)
subject_list = np.array(subject_list)
y = np.array(y)

from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Masking, LSTM, Bidirectional, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
import json

# パラメータ設定
input_dim = len(new_columns) - 1  # 入力データの次元数
num_classes = 4                  # クラス数
len_sequence = max_len_sequence  # 時系列の長さ
batch_size = 64                  # ミニバッチサイズ
epochs = 50                      # 学習エポック数
padding_value = -9999              # Maskingで使用する値

# モデル評価用
scores = []
models = []

# GroupKFoldの設定
group_kfold = GroupKFold(n_splits=len(set(subject_list)))

now = time.localtime()
str_now = time.strftime('%Y%m%d%H%M%S', now)
save_path = f"../../Result/OpenData/log_{str_now}_LSTM.csv"
# f = open(f"../../Result/OpenData/log_{str_now}.txt", 'w')

acc = []
y_voting_res = []
subject_y = []
pred_y_list = []
new_y = []
scores = []
models = []

# クロスバリデーション
for fold, (train_index, test_index) in enumerate(group_kfold.split(X, y, subject_list)):
    X_train = X_padded[train_index]
    y_train = y[train_index]
    X_test = X_padded[test_index]
    y_test = y[test_index]

    # LSTM モデルの入力テンソルを定義
    lstm_input = Input(shape=(len_sequence, input_dim))
    lstm_model = Masking(mask_value=padding_value)(lstm_input)
    lstm_model = Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.5))(lstm_model)
    lstm_output = lstm_model

    # 最終モデル
    model = Model(inputs=lstm_input, outputs=lstm_output)

    # モデルコンパイル
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    # コールバック設定
    rlr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=3,
                            verbose=1,
                            min_delta=1e-3,
                            mode='min')

    es = EarlyStopping(monitor='val_loss',
                       patience=10,
                       mode='min',
                       restore_best_weights=True,
                       verbose=1)

    # モデルの学習
    hist = model.fit(X_train, y_train,  # CNN用のデータは不要なので削除
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=[rlr, es],
                     validation_data=(X_test, y_test))

    print(f"Fold {fold + 1} training finished!")

    # テストデータで評価
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold + 1} - Test loss: {score[0]}, Test accuracy: {score[1]}")

    y_pred_score = model.predict(X_test)
    y_pred = np.argmax(y_pred_score, axis=1)
    # 予測されたクラスの多数決を行う
    y_pred_counter = Counter(y_pred)
    most_common_count = y_pred_counter.most_common()[0][1]
    majority_class = y_pred_counter.most_common()[0][0]

    # 同数のクラスがあるかを確認
    if len(y_pred_counter.most_common()) > 1 and (most_common_count == y_pred_counter.most_common()[1][1]):
        # 同数のクラスをリストに追加
        majority_classes = [majority_class]
        for i in range(1, len(y_pred_counter.most_common())):
            if most_common_count == y_pred_counter.most_common()[i][1]:
                majority_classes.append(y_pred_counter.most_common()[i][0])
            else:
                break

        # そのクラスとして予測されたインデックスを取得
        indices = [i for i, value in enumerate(y_pred) if value in majority_classes]

        # 各クラスの予測確率を取得
        proba = y_pred_score
        # f.write(f"予測確率: {proba[indices]}\n")

        # 各クラスに対して確率の最大を取得
        class_probs = {}
        for majority_class in majority_classes:
            class_indices = [j for j, value in enumerate(y_pred) if value == majority_class]
            class_prob_max = np.max([proba[j][majority_class] for j in class_indices])
            # class_prob_avg = np.mean([proba[j][majority_class] for j in class_indices])　# 各クラスの予測確率の平均で比較する場合
            class_probs[majority_class] = class_prob_max

        # 最も確率が高いクラスを選択
        voting_res = max(class_probs, key=class_probs.get)

        # f.write(f"各クラス {majority_classes} の最大確率: {class_probs}\n")
        # f.write(f"選ばれたクラス: {voting_res}\n\n")
    else:
        # 明確な多数決がある場合、そのクラスを選択
        voting_res = majority_class
    # 結果を保存
    subject_y.append(y_test[0])
    y_voting_res.append(voting_res)

    # 精度を計算
    fold_acc = accuracy_score([y_test[0]], [voting_res])
    acc.append(fold_acc)
    
    # **モデルの保存**
    model.save(f'./model/model_fold{fold+1}.h5')  # 学習済みモデルの重みを保存

    with open(f'./model/model_fold{fold+1}.json', 'w') as f:
        f.write(model.to_json())  # モデルの構造を保存

    # **ハイパーパラメータの保存**
    hyperparams = {
        "Fold": fold + 1,
        "LSTM_units": 128,
        "Dropout_rate": 0.5,
        "Recurrent_dropout": 0.5,
        "Batch_size": batch_size,
        "Epochs": epochs,
        "Optimizer": "Adam",
        "Loss": "sparse_categorical_crossentropy"
    }

    with open(f'./model/hyperparams_fold{fold+1}.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)  # ハイパーパラメータをJSONで保存
        
    # CSVに保存
    df = pd.DataFrame([[fold + 1, score[0], score[1], fold_acc]],
                      columns=["Fold", "Test Loss", "Test Accuracy", "Majority Voting Accuracy"],
                      dtype=object)

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)

    scores.append(score[1])  # 精度を保存
    models.append(model)     # モデルを保存

# クロスバリデーションの平均スコア
cv_score = np.mean(scores)
print(f'CV score: {cv_score}')
print(f'Subject score: {np.mean(acc)}')
