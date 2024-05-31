import os
import string
import random
import math
import numpy as np
import torch
from datetime import datetime


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