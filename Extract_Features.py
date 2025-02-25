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


from scipy import signal

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
        
def convert_vec(data_x,data_y,data_z):
    return np.sqrt(
        data_x ** 2 + data_y ** 2 + data_z ** 2)

from pyampd.ampd import find_peaks
def getargextrema(data,order):
    maxId=[]
    minId=[]

    maxId = find_peaks(data,order)
    minId = find_peaks(-data,order)

    threshold = np.mean(data)
    
    maxId = maxId[data[maxId]>threshold]
    minId = minId[data[minId]<threshold]
    
    i=0
    j=0
    maximaId = [maxId[i]]
    minimaId = [minId[i]]
    #
    while i < len(maxId) and j < len(minId):
        if maximaId[-1] < minimaId[-1]:
            count = len(maxId[(maximaId[-1] < maxId) & (maxId < minimaId[-1])])
            while count:
                count -= 1
                i += 1
                if data[maximaId[-1]] < data[maxId[i]]:
                    maximaId[-1] = maxId[i]
            i += 1
            if i < len(maxId):
                maximaId.append(maxId[i])
        elif maximaId[-1] > minimaId[-1]:
            count = len(minId[(maximaId[-1] > minId) & (minId > minimaId[-1])])
            while count:
                count -= 1
                j += 1
                if data[minimaId[-1]] > data[minId[j]]:
                    minimaId[-1] = minId[j]
            j += 1
            if j < len(minId):
                minimaId.append(minId[j])
    if maximaId[0] > minimaId[0]:
      minimaId.pop(0)
    if maximaId[-1] > minimaId[-1]:
      maximaId.remove(maximaId[-1])

    return maximaId,minimaId

def diffentiation_signal(data):
    dy = np.gradient(data)
    return dy

def fourier_tarnsform(data,dt=1.0/200):
    F = np.fft.fft(data)
    N = len(data)
    freq = np.fft.fftfreq(N, d=dt)
 
    Amp = np.abs(F / (N / 2))
    return freq,Amp


# Exsiting Features
def minja_features(data):
    rms = np.sqrt(np.mean(data**2))
    minimum = np.min(data)
    maximum = np.max(data)
    avg = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    
    freq,Amp = fourier_tarnsform(data)
    freq = freq[:len(freq)//2+1]
    Amp = Amp[:len(Amp)//2+1]
    
    max_freq = freq[np.argmax(Amp)]
    centroid = np.sum(Amp*freq)/np.sum(Amp)
    
    order=int(len(data)*0.05)
    peaks, _ = getargextrema(data,order)
    print(len(peaks))
    rms_of_max_taps = np.sqrt(np.mean(data[peaks]**2))
    min_of_max_taps = np.min(data[peaks])
    max_of_max_taps = np.max(data[peaks])
    avg_of_max_taps = np.mean(data[peaks])
    std_of_max_taps = np.std(data[peaks])
    mid_of_max_taps = np.median(data[peaks])
    
    features = [rms,minimum,maximum,avg,std,median,max_freq,centroid,
                rms_of_max_taps,min_of_max_taps,max_of_max_taps,avg_of_max_taps,std_of_max_taps,mid_of_max_taps
               ]
    return features

import numpy as np
import pandas as pd
import math
import scipy
from scipy import optimize
# Newly Features
def noise_variance_estimation_for_1D(signal:np.ndarray((1,),dtype=np.float64)):
    if signal.ndim != 1:
        raise Exception('This function is for 1D signal. If you want to used NVE for 2 or more than dimensions data, you used other method')

    signal_size = np.array([1,signal.size],dtype=int)#次元の長さ
    signal_ndim=2#次元数

    S = np.zeros(signal_size)

    for i in range(signal_ndim):
        siz0 = np.ones(signal_ndim,dtype=int)
        siz0[i] = signal_size[i]
        _data = np.arange(1,signal_size[i]+1)
        _data = np.reshape(_data,siz0)-1
        _data = np.pi * _data
        _data = _data/signal_size[i]
        data = np.cos(_data)
        S = S + data


    S = 2*(signal_ndim-S)


    y = dctn_for_1D(signal)
    y = np.reshape(y,(-1,1))

    S = np.power(S,2)
    y = np.power(y,2)

    N = 1
    hMin = 1e-6
    hMax = 0.99
    sMinBnd = (math.pow(((1+math.sqrt(1+8*math.pow(hMax,(2/N))))/4/math.pow(hMax,(2/N))),2)-1)/16
    sMaxBnd = (math.pow(((1+math.sqrt(1+8*math.pow(hMin,(2/N))))/4/math.pow(hMin,(2/N))),2)-1)/16


    def func(L):
        M = (1-(1/(1+(np.power(10,L)*S))))
        M = np.reshape(M,(-1,1))
        noisevar = np.mean(y*np.power(M,2))
        return noisevar/np.power(np.mean(M),2)

    minimum = optimize.fminbound(func,math.log10(sMinBnd),math.log10(sMaxBnd),xtol=0.1)
    M = 1 -1/(1+np.power(10,minimum)*S)
    M = np.reshape(M,(-1,1))
    noisevar = np.mean(y*np.power(M,2))

    return [noisevar]

def dctn_for_1D(y:np.ndarray((1,),dtype=np.float64)):
    if y.ndim != 1:
        raise Exception('This function is for 1D signal. If you want to used NVE for 2 or more than dimensions data, you used other method')

    y_size = np.array([1,y.size],dtype=int)
    y = np.squeeze(y)
    y_dim=1
    y = np.reshape(y,(-1,1))
    n = y.size

    _w = np.arange(0,n)
    _w = np.reshape(1j*_w*np.pi,(-1,1))
    _w = _w/2/n
    w = np.exp(_w)


    siz = np.array([y.size,1])
    n = siz[0]
    _y_index1 = list(range(1,n+1,2))
    _y_index2 = list(range(2*math.floor(n/2),1,-2))

    y_index = np.array(_y_index1+_y_index2,dtype=int)-1

    y= y[y_index]
    y = np.reshape(y,(n,-1))

    y = y*math.sqrt(2*n)

    y = np.fft.ifft(y,axis=0)
    y=y*w

    y = np.real(y)

    y[0] = y[0]/math.sqrt(2)

    y = np.reshape(y,siz)
    y = np.reshape(y,(1,-1))
    y = np.reshape(y,y_size)
    return y

def nr_features(data):
    conv_ene=0
    for d in data:
      conv_ene += d*d
    noise = noise_variance_estimation_for_1D(data)
    snr = float((conv_ene/len(data))/noise)
    features = [conv_ene,snr]
    return features

import matplotlib.pylab as plt
import seaborn as sns
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

def gettsfresh_features(df,explanatory,response):
  id = pd.DataFrame({"id":[1]*len(df)})
  new_df = pd.concat([id,df[[explanatory,response]]],axis=1,join="outer")
  fc_parameters = {
        "variance": None,
        "mean_abs_change": None,
        "autocorrelation": [{"lag":i} for i in range(1,10)],
        "quantile":[{"q":i*0.1} for i in [1,2,3,4,6,7,8,9]],
  }
    
  X = extract_features(new_df,column_id = "id",column_sort=explanatory,n_jobs = 1, default_fc_parameters=fc_parameters)
  return X.values.tolist()[0]

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

from scipy.signal import argrelextrema, butter, sosfilt
def new_feature_extraction(data,column):
    df = data
    # feature variable initialization
    rhythm, mx_mx, mx_mn, mx_av, mx_std = [], [], [], [], []
    mn_mx, mn_mn, mn_av, mn_std, amp = [], [], [], [], []
    freq, slop, velocity, velstd, freqstd = [], [], [], [], []
    task_parameters_norm = normalize(data, 0, 100)
    filtered_signal = sm.tsa.filters.bkfilter(task_parameters_norm)
    x = np.array(filtered_signal)
    (f, S) = scipy.signal.periodogram(x, 128, scaling='density')
    # r=np.max(S)
    r = np.sqrt(np.mean(S ** 2))
    rhythm.append(r)
    # rythom=np.sqrt(np.mean(S**2))
    order = 1
    local_maximal_index = argrelextrema(x, np.greater_equal, order=order)
    local_maximal_index = (np.squeeze(local_maximal_index))
    # print(local_maximal_index)
    local_maximal_value = x[argrelextrema(x, np.greater_equal, order=order)[0]]
    # for local minima
    local_minimal_index = argrelextrema(x, np.less_equal, order=order)
    local_minimal_index = (np.squeeze(local_minimal_index))
    local_minimal_value = x[argrelextrema(x, np.less_equal, order=order)[0]]

    # amplitude
    l = min(len(local_maximal_value), len(local_minimal_value))
    am = (local_maximal_value[:l] - local_minimal_value[:l]).mean()
    amp.append(am)
    # slope  https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:linear-equations-graphs/x2f8bb11595b61c86:slope/v/slope-from-table
    x = np.array(task_parameters_norm)
    y = [y for y in range(len(x))]
    val_x = sm.add_constant(y)
    val_y = x
    output = sm.OLS(val_y, val_x).fit()
    
    sl = output.params[1]
    slop.append(sl)
    
    try:
        t = [t for t in range(len(local_maximal_index))]
        f = 1 / (np.mean(np.diff(local_maximal_index)))
    except:
        f = 0
    freq.append(f)  # freequency

    try:
        fstd = 1 / (np.std(np.diff(local_maximal_index)))
    except:
        fstd = 0
    freqstd.append(fstd)
    
    return [rhythm[0],amp[0],freq[0],freqstd[0],slop[0]]

new_dfs = []
new_columns = ["Thumb_x_vel","Thumb_y_vel","Thumb_z_vel","Index_x_vel","Index_y_vel","Index_z_vel",
            "Thumb_vec_vel","Index_vec_vel","Thumb2Index_vec_vel",
            "Thumb_x_acc","Thumb_y_acc","Thumb_z_acc","Thumb_vec_acc",
            "Index_x_acc","Index_y_acc","Index_z_acc","Index_vec_acc",
            "Thumb2Index_vec_acc",
               "Timestamp"
              ]

remove_dfs = []
new_target_list = []
out_target_list = []
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

    new_signal.extend([Thumb_x_acc,Thumb_y_acc,Thumb_z_acc,Thumb_vec_acc,
                   Index_x_acc,Index_y_acc,Index_z_acc,Index_vec_acc,
                   Thumb2Index_vec_acc])
    
    timestamp = np.array(list(range(len(Thumb_x)))) * 0.005 #200Hz
    new_signal.extend([timestamp])
    
    new_signal = np.array(new_signal)
    new_df = pd.DataFrame(new_signal.T,columns=new_columns)

    new_dfs.append(new_df)
    new_target_list.append(target_list[i])
f = open('../../ExtractedFeatures/OpenData/outlier_log.txt', 'w')
f.write(f"# of Remove data:{len(out_target_list)}\n")
f.write(f"Remove files:{out_target_list}\n")
f.close()

feature_list = []
label_list = []
for i,df in enumerate(new_dfs):
    print(f"{i}:{new_target_list[i]}")
    features = []
    
    for column in df.columns.values[:-1]:
        print(column)
        signal = df[column].values
        
        minja_feature = minja_features(signal)
        noise_feature = noise_variance_estimation_for_1D(signal)
        noise_ratio_feature = nr_features(signal)
        tsfresh_feature = gettsfresh_features(df, "Timestamp", column)
        musa_faeture = new_feature_extraction(signal,column)
        
        features += minja_feature  + noise_feature + noise_ratio_feature + tsfresh_feature + musa_faeture
        # features += musa_faeture + minja_feature
    
    disease,subject_id,_ = new_target_list[i].split('_')
    label = []
    if disease == "CTRL":
        label = [0]
    elif disease == "PD":
        label = [1]
    elif disease == "PSP":
        label = [2]
    elif disease == "MSA":
        label = [3]
    
    feature_list.append(features+[subject_id]+label)
    label_list.append(label[0])

columns = new_dfs[0].columns.values 
feature_names = []
for signal in columns[:-1]:
    
    feature_names += [signal+"_RMS",signal+"_min",signal+"_max",signal+"_avg",signal+"_std",signal+"_median",signal+"_max_freq",signal+"_centroid",
                      signal+"_RMS_of_max_taps",signal+"_min_of_max_taps",signal+"_max_of_max_taps",signal+"_avg_of_max_taps",signal+"_std_of_max_taps",signal+"_median_of_max_taps",
                 signal+"_noise_var",
                 signal+"_conv_ene",signal+"_SNR",
                 signal+"_variance",signal+"_mean_abs_change"]+[signal+"_autocorrelation_lag_"+str(i) for i in range(1,10)]+[signal+"_quantile_q_0."+str(i) for i in range(1,5)]+[signal+"_quantile_q_0."+str(i) for i in range(6,10)]
    
    feature_names += [signal+'_rhythm',
                      column+'_mx-mx', column+'_mx-mn', column+'_mx-av', column+'_mx-std', column+'_mn-mx',column+'_mn-mn', column+'_mn-av', column+'_mn-std',
                      signal+'_amplitude', signal+'_frequency', signal+'_freqstd',
                      signal+'_slop', 
                      column+'_velocity', column+'_velstd'
                     ]
        
    
def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
        
dir_path = "../../ExtractedFeatures/OpenData/"
my_makedirs(dir_path)
fname = "new_FT_4class.csv"
fpath = os.path.join(dir_path, fname)
subject_status = ["subjectid", "class"]
feature_names +=subject_status
output = pd.DataFrame(feature_list, columns=feature_names)
output.to_csv(fpath, mode="w",index=False)