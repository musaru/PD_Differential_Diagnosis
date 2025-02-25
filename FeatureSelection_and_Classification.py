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
from collections import Counter

path = '~/Parkisonism_Diagnosis/ExtractedFeatures/OpenData/new_FT_4class.csv'

df = pd.read_csv(os.path.join(path))

df = df.dropna(axis=1)

df_HC = df[df["class"] == 0]
df_PD = df[df["class"] == 1]
df_PSP = df[df["class"] == 2]
df_MSA = df[df["class"] == 3]
y = np.concatenate([df_HC["class"].values ,df_PD["class"].values ,df_PSP["class"].values ,df_MSA["class"].values])
df_subject_list = pd.concat([df_HC.iloc[:,-2:] ,df_PD.iloc[:,-2:] ,df_PSP.iloc[:,-2:] ,df_MSA.iloc[:,-2:]])

X_HC = df_HC.values[:,:-2].astype(np.float64)
X_PD = df_PD.values[:,:-2].astype(np.float64)
X_PSP = df_PSP.values[:,:-2].astype(np.float64)
X_MSA = df_MSA.values[:,:-2].astype(np.float64)
print(len(X_HC))
print(len(X_PD))
print(len(X_PSP))
print(len(X_MSA))
print(y)
print(len(y))

subject_list = []
for class_label,subject in zip(df_subject_list["class"],df_subject_list["subjectid"]):
    if class_label == 0:
        disease = "HC"
    elif class_label == 1:
        disease = "PD"
    elif class_label == 2:
        disease = "PSP"
    elif class_label == 3:
        disease = "MSA"
        
    subject_info = f"{disease}_{subject}"
    subject_list.append(subject_info)
subject_list = np.array(subject_list)
features_name_list = df.columns.values[:-2]

# feature selection
X_HC = np.array(X_HC)
X_PD = np.array(X_PD)
X_PSP = np.array(X_PSP)
X_MSA = np.array(X_MSA)

alpha = 0.005
selected_features = []
selected_features_index = []
p_values = []
for i,feature in enumerate(features_name_list):
    anova_result = f_oneway(X_HC[:,i],X_PD[:,i],X_PSP[:,i],X_MSA[:,i])
    if anova_result.pvalue < alpha: #
        p_values.append(anova_result.pvalue)
        selected_features.append(feature)
        selected_features_index.append(i)

p_values = np.array(p_values)
selected_features = np.array(selected_features)
selected_features_index = np.array(selected_features_index)
        
selected_features = selected_features[np.argsort(p_values)]
selected_features_index = selected_features_index[np.argsort(p_values)]
p_values = np.sort(p_values)

X = np.concatenate([X_HC,X_PD,X_PSP,X_MSA])
#label: 0,1,2,3
based_feature_index  = []
used_feature_index = []
best_used_feature_index=[]

# initial accuracy (threshold)
goal_accuracy = 0.86

now = time.localtime()
str_now = time.strftime('%Y%m%d%H%M%S', now)
save_path1 = f"../../Result/OpenData/all_log_{str_now}_SVM.csv"
save_path2 = f"../../Result/OpenData/best_log_{str_now}_SVM.csv"

accuracy_hist = []
used_feature_index_hist = []

max_acc = -1
while(max_acc < goal_accuracy):
    best_param=None
    max_acc = -1
    for i in range(len(selected_features_index)):
        if selected_features_index[i] in based_feature_index:
            continue
        used_feature_index = based_feature_index + [selected_features_index[i]]

        print(used_feature_index)
        
        def objective(trial):
            kernel = trial.suggest_categorical('kernal', [ 'rbf', 'sigmoid'])
            
            svc_c = trial.suggest_float("SVC_C", 0.01, 100)        
            svc_gamma = trial.suggest_float("SVC_gamma", 0.01, 100)
            classifier = SVC(
                    kernel=kernel, C=svc_c, gamma=svc_gamma, random_state=42
                    )

            score_funcs = [
            'accuracy',
            ]
            
            steps = list()
            
            steps.append(('scaler', MinMaxScaler()))
            steps.append(('model', classifier))
            pipeline = Pipeline(steps=steps)
            
            group_kfold = GroupKFold(n_splits=len(set(subject_list)))
            
            acc = []
            y_voting_res = []
            subject_y = []
            pred_y_list = []
            new_y = []

            # Make predictions for each fold
            for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, subject_list)):
                # Prepare training and test data
                X_train = X[:, used_feature_index][train_index]
                y_train = y[train_index]
                X_test = X[:, used_feature_index][test_index]
                y_test = y[test_index]

                # Model training and prediction
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Save predictions of data
                pred_y_list.append(y_pred)

                # Majority vote
                y_pred_counter = Counter(y_pred)
                most_common_count = y_pred_counter.most_common()[0][1]
                majority_class = y_pred_counter.most_common()[0][0]

                #  Check if there is an equal number of classes
                if len(y_pred_counter.most_common()) > 1 and (most_common_count == y_pred_counter.most_common()[1][1]):
                    # Add equal number of classes to list
                    majority_classes = [majority_class]
                    for i in range(1, len(y_pred_counter.most_common())):
                        if most_common_count == y_pred_counter.most_common()[i][1]:
                            majority_classes.append(y_pred_counter.most_common()[i][0])
                        else:
                            break

                    # Get the index predicted as their class(majority_class)
                    indices = [i for i, value in enumerate(y_pred) if value in majority_classes]

                    # Get the prediction probability for each class
                    proba = np.array(pipeline.decision_function(X_test))

                    # Get the maximum probability for each class
                    class_probs = {}
                    for majority_class in majority_classes:
                        class_indices = [j for j, value in enumerate(y_pred) if value == majority_class]
                        class_prob_max = np.max([proba[j][majority_class] for j in class_indices])
                        class_probs[majority_class] = class_prob_max

                    # The class with the highest probability is the patient's prediction.
                    voting_res = max(class_probs, key=class_probs.get)

                    
                else:
                    # If there is no class with the same number, select that class
                    voting_res = majority_class

                # Save predictions of patient
                subject_y.append(y_test[0])
                y_voting_res.append(voting_res)
                
                # Calcurate accuracy
                acc.append(accuracy_score([y_test[0]], [voting_res]))

            return np.mean(acc) 
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=200,gc_after_trial=True)
        
        if max_acc < study.best_value:
            max_acc = study.best_value
            best_used_feature_index = copy.deepcopy(used_feature_index)
            best_param = study.best_params
            
        df = pd.DataFrame(np.array([[study.best_value,used_feature_index,study.best_params]],dtype=object))
        if os.path.exists(save_path1):
            df.to_csv(save_path1,mode='a',header=False)
        else:
            df.to_csv(save_path1,mode='w')
    

    if len(best_used_feature_index) >= 2:
        for idx in best_used_feature_index:
            used_feature_index = [f for f in best_used_feature_index if f != idx]            
            print("Remove Phase: ")
        
            study = optuna.create_study(direction='maximize')
            study.optimize(objective,n_trials=200,gc_after_trial=True)

            if max_acc <= study.best_value:
                max_acc = study.best_value
                best_used_feature_index = copy.deepcopy(used_feature_index)
                best_param = study.best_params

            df = pd.DataFrame(np.array([[study.best_value,used_feature_index,study.best_params]],dtype=object))
            if os.path.exists(save_path1):
                df.to_csv(save_path1, mode='a', header=False)
            else:
                df.to_csv(save_path1, mode='w')

    based_feature_index = copy.deepcopy(best_used_feature_index)

    df = pd.DataFrame(np.array([[max_acc, based_feature_index, best_param]], dtype=object))    
    if os.path.exists(save_path2):
        df.to_csv(save_path2, mode='a', header=False)
    else:
        df.to_csv(save_path2, mode='w')