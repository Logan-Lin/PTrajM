import os
import string
import random
import math
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, accuracy_score, roc_auc_score


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)


def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def create_if_noexists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def cal_courseAngle(lng1, lat1, lng2, lat2):
#     y = math.sin(lng2-lng1) * math.cos(lat2)
#     x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lng2-lng1)
#     bearing = math.atan2(y, x)
#     bearing = 180 * bearing / math.pi
#     if bearing < 0:
#         bearing = bearing + 360
#     return bearing
def cal_courseAngle(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    y = np.sin(lng2-lng1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng2-lng1)
    bearing = np.arctan2(y, x)
    bearing = 180 * bearing / np.pi
    bearing = np.where(bearing < 0, bearing + 360, bearing)
    return bearing

def cal_geo_distance(lng1, lat1, lng2, lat2):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    return distance

def cal_tensor_geo_distance(lng1:torch.tensor, lat1:torch.tensor, lng2:torch.tensor, lat2:torch.tensor):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    distance = 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000 # + 1e-8——不能在a后加，出大问题坏！
    return distance

def cal_tensor_courseAngle(lng1:torch.tensor, lat1:torch.tensor, lng2:torch.tensor, lat2:torch.tensor):
    lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])
    y = torch.sin(lng2-lng1) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(lng2-lng1)
    bearing = torch.arctan2(y, x)
    bearing = 180 * bearing / torch.pi
    bearing = torch.where(bearing < 0, bearing + 360, bearing)
    return bearing


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value
    

def mean_absolute_percentage_error(y_true, y_pred):
    """ Calculcates the MAPE metric. """
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape

def cal_regression_metric(label, pres):
    """ Calculcate all common regression metrics. """
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)

    s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
    return s


def distance_mae(distance, null_val=np.nan):
    distance_mae = np.mean(np.abs(distance))
    return distance_mae
    # if np.isnan(null_val):
    #     mask = ~torch.isnan(distance)
    # else:
    #     mask = (distance!=null_val)
    # mask = mask.float()
    # mask /=  np.mean((mask))
    # mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    # # loss = torch.abs(preds-labels)
    # loss = np.abs(distance)
    # loss = loss * mask
    # loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    # return np.mean(loss)

def distance_mse(distance, null_val=np.nan):
    distance_mse = np.mean(distance**2)
    return distance_mse
    # if np.isnan(null_val):
    #     mask = ~torch.isnan(distance)
    # else:
    #     mask = (distance!=null_val)
    # mask = mask.float()
    # mask /= np.mean((mask))
    # mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    # # loss = (preds-labels)**2
    # loss = distance**2
    # loss = loss * mask
    # loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    # return np.mean(loss)

def distance_rmse(distance, null_val=np.nan):
    return np.sqrt(distance_mse(distance=distance, null_val=null_val))

def cal_distance_metric(label, pres, lng_col, lat_col):
    """ 
    Calculcate all distance regression metrics. 

    :param labels: longitude and latitude features of the trajectories, with shape (B,2).
    :param pres: predicted longitude and latitude features of the trajectories, with shape (B, 2).
    """
    distance = cal_geo_distance(label[...,lng_col], label[...,lat_col], pres[...,lng_col], pres[...,lat_col])
    mae = distance_mae(distance, 0.0)#.item()
    rmse = distance_rmse(distance, 0.0)#.item()
    s = pd.Series([rmse, mae], index=['distance_rmse', 'distance_mae'])
    return s


def top_n_accuracy(truths, preds, n):
    """ Calculcate Acc@N metric. """
    # best_n = np.argsort(preds, axis=1)[:, -n:] # 升序排列求后n个
    best_n = np.argsort(-preds, axis=1)[:, :n] # 降序排列求前n个
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classification_metric(labels, pres):
    """
    Calculates all common classification metrics.

    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    pres_index = pres.argmax(-1)  # (N)
    macro_f1 = f1_score(labels, pres_index, average='macro', zero_division=0)
    macro_recall = recall_score(labels, pres_index, average='macro', zero_division=0)
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]

    s = pd.Series([macro_f1, macro_recall, acc] + top_n_acc,
                  index=['macro_f1', 'macro_rec'] +
                  [f'acc@{n}' for n in [1] + n_list])
    return s