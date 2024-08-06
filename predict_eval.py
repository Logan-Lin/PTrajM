import os
import json
import numpy as np
from argparse import ArgumentParser
from functools import partial
from data import X_COL, Y_COL, DT_COL, ROAD_COL
from utils import cal_classification_metric, cal_regression_metric, cal_distance_metric

PRED_SAVE_DIR = os.environ.get('PRED_SAVE_DIR', 'predictions')

parser = ArgumentParser()
parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test_lnglat") # required=True
args = parser.parse_args()

# Load the settings file, and save a backup in the cache directory.
with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
    settings = json.load(fp)

# Iterate through the multiple settings.
for setting_i, setting in enumerate(settings):
    print(f'===SETTING {setting_i}/{len(settings)}===')
    SAVE_NAME = setting.get('save_name', None)
    
    if 'test' in setting:
        test_setting = setting['test']
        if test_setting.get('save', False):
            predictions = np.load(os.path.join(PRED_SAVE_DIR, SAVE_NAME, 'predictions.npy'))
            targets = np.load(os.path.join(PRED_SAVE_DIR, SAVE_NAME, 'targets.npy'))
            print(f"predictions: {predictions.shape}")
            # print(f"targets: {targets.shape}")
            
            pred_cols = test_setting['padder']['params']["pred_cols"] # list
            if sorted(pred_cols) == sorted([Y_COL, X_COL]): # 预测'lng''lat'
                lng_col, lat_col = pred_cols.index(X_COL), pred_cols.index(Y_COL)
                # print(f"lng_col: {lng_col}, lat_col: {lat_col}")
                cal_metric = partial(cal_distance_metric, lng_col=lng_col, lat_col=lat_col)
                metric_filename = "gps_regression"
            elif pred_cols == [DT_COL]: # 预测'delta_t'
                cal_metric = cal_regression_metric
                metric_filename = "delta_t_regression"
            else:
                raise NotImplementedError(f'No predict columns called "{pred_cols}".')
                
            metric = cal_metric(targets, predictions)
            print(f"the test metric for {pred_cols}:")
            print(metric)
            metric.to_hdf(os.path.join(PRED_SAVE_DIR, SAVE_NAME, f'{metric_filename}.h5'), key='metric', format='table')