#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:32:47 2022

@author: guohan
"""

import os, sys, warnings

path_list = sys.path
module_path = '/Users/guohan/Documents/Codes/Regression_Analysis'
if module_path not in sys.path:
    sys.path.append(module_path)
    print('Add module path')

from main import *


if __name__ == '__main__':

    ### Combine results ###
    srcDir = 'tests/prediction'
    input_file_target = 'tests/target.csv'
    output_file = 'combination'
    combine_multiple_expts(srcDir, input_file_target, output_file)

    ### Recovery rate ###
    prediction_file = 'tests/combination_target.csv'
    target_file = 'tests/target.csv'
    ratio_top_prediction_list = [0.02, 0.04, 0.05, 0.10]
    ratio_top_target_list = [0.02, 0.04, 0.05, 0.10]
    prediction_column_name_list = ['a', 'expt0', 'expt1']
    RR_matrix = get_recovery_rate(prediction_file, target_file, ratio_top_prediction_list, ratio_top_target_list, prediction_column_name_list)

    # write recovery rate to .csv output_file
    df = pd.DataFrame(dict(zip(prediction_column_name_list, RR_matrix)))
    df.to_csv('Recovery_rate.csv')

    ### Select data based on rules ###
    input_file = 'test/test_get_trainset.csv'
    current_trainset = 'test/target.csv'
    ratio_top_ave = 0.1
    ratio_top_std = 0.05
    ratio_random = 0.01

    get_active_learning_trainset(input_file, current_trainset, ratio_top_ave, ratio_top_std, ratio_random)