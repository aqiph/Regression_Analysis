#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:09:28 2022

@author: guohan

"""

import os
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_size(12)



### Combine results ###

def combine_multiple_expts(srcDir, input_file_target = None, output_file = None):
    """
    combine predicted results and true labels
    :param srcDir: str, directory path of predicted results
    :param input_file_target: str, path of the file containing true labels
    :param output_file: str, name of the output file
    :return: int, number of compounds
    """
    num = 0
    expts = []  # experiment names
    dfs = []

    # read df
    files = os.listdir(srcDir)
    for file in files:
        if os.path.splitext(file)[1] != '.csv':
            continue

        try:
            expt = os.path.splitext(file)[0]
            expts.append(expt)
            df = pd.read_csv(os.path.join(srcDir, file))
            df = pd.DataFrame(df, columns=['ID', 'Cleaned_SMILES', 'score'])
            df.rename(columns={'score': expt}, inplace=True)
            dfs.append(df)
            print('{} is read, number of rows: {}'.format(expt, df.shape[0]))
            num += 1

        except:
            pass

    # output file
    folder, basename = os.path.split(os.path.abspath(srcDir))
    if output_file is None:
        output_file = basename
    output_file = os.path.join(folder, os.path.splitext(output_file)[0])

    # merge multiple experiments
    df_merge = reduce(lambda left, right: pd.merge(left, right, how='outer', on=['ID', 'Cleaned_SMILES']), dfs)
    columns = ['ID', 'Cleaned_SMILES'] + sorted(expts)
    df_merge = df_merge.reindex(columns, axis=1)

    # compute average and std
    df_merge['Ave_Score'] = df_merge.apply(lambda row: np.round(np.mean([row[expt] for expt in expts]), 16), axis=1)
    df_merge['Uncertainty'] = df_merge.apply(lambda row: np.round(np.std([row[expt] for expt in expts]), 15), axis=1)

    # merge target labels
    if input_file_target is not None:
        df_target = pd.read_csv(input_file_target)
        df_target = pd.DataFrame(df_target, columns=['ID', 'Cleaned_SMILES', 'Label'])
        df_target.rename(columns={'ID': 'Training_set_ID', 'Label': 'True_Label'}, inplace=True)
        df_merge = pd.merge(df_merge, df_target, how='left', on=['Cleaned_SMILES'])
    
    # write to file
    df_merge.sort_values(by = ['Ave_Score'], ascending = True, inplace = True)
    print('Number of rows in the output file:', df_merge.shape[0])
    df_merge.to_csv(output_file + f'_{df_merge.shape[0]}.csv')

    return df_merge.shape[0]


### Recovery rate ###

def get_recovery_rate(prediction_file, target_file, ratio_top_prediction_list, ratio_top_target_list,
                      prediction_column_name_list = ['Ave_Score'], target_column_name = 'Label'):
    """
    Compute the recovery rate
    :param prediction_file: str, the name of the prediction file
    :param target_file: str, the name of the target file
    :param ratio_top_prediction_list: list of floats, ratios of top scoring predicted compounds defined as predicted active
    :param ratio_top_target_list: list of floats, ratios of top scoring target compounds defined as virtual hit compounds
    :param prediction_column_name_list: list of strs, list of column names of the predicted scores used to compute recovery rate
    :param target_column_name: str, column name of the target score used to compute recovery rate
    :return: np.matrix, recovery rate matrix
    """
    # read files
    df_prediction = pd.read_csv(prediction_file, index_col=0)
    df_target = pd.read_csv(target_file, index_col=0)

    num_prediction, num_target = df_prediction.shape[0], df_target.shape[0]
    num_prediction_active_list = [int(np.around(num_prediction * ratio_top_prediction)) for ratio_top_prediction in ratio_top_prediction_list]
    num_target_active_list = [int(np.around(num_target * ratio_top_target)) for ratio_top_target in ratio_top_target_list]

    # compute recovery rate
    num_ratio = len(ratio_top_prediction_list)
    assert len(ratio_top_prediction_list) == len(ratio_top_target_list), 'Error: different lengths of ratio_top_prediction_list and ratio_top_target_list'
    num_predictions = len(prediction_column_name_list)
    RR_matrix = np.zeros((num_predictions, num_ratio), dtype=np.float16)

    for i, prediction_column_name in enumerate(prediction_column_name_list):
        df_prediction.sort_values(by=[prediction_column_name], ascending=True, inplace=True)
        df_target.sort_values(by=[target_column_name], ascending=True, inplace=True)

        for j, num_prediction_active in enumerate(num_prediction_active_list):
            df_prediction_top = df_prediction.iloc[:num_prediction_active]
            df_target_top = df_target.iloc[:num_target_active_list[j]]
            df_intersection = pd.merge(df_prediction_top, df_target_top, how='inner', on=['Cleaned_SMILES'])

            if False:
                folder, basename = os.path.split(os.path.abspath(input_file))
                print('Number of true positive is {}'.format(df_intersection.shape[0]))
                df_intersection.to_csv(os.path.join(folder, 'True_Positive.csv'))
                df_prediction_top.to_csv(os.path.join(folder, 'prediction_positive.csv'))
                df_target_top.to_csv(os.path.join(folder, 'target_positive.csv'))

            recall = float(df_intersection.shape[0]) / float(df_target_top.shape[0])
            precision = float(df_intersection.shape[0]) / float(df_prediction_top.shape[0])
            RR_matrix[i, j] = recall
            print('{} @ {} Recall is {:4.4f} precision is {:4.4f}\n'.format(prediction_column_name, ratio_top_prediction_list[j], recall, precision))

    # plot distribution
    fig, ax = plt.subplots()
    x = np.arange(num_ratio)
    width = 1.0/(num_predictions + 2)

    for i in range(num_predictions):
        plt.bar(x + (i+0.5-num_predictions/2) * width, RR_matrix[i, :], width = width, label = prediction_column_name_list[i])

    plt.ylabel('Recovery Rate')
    plt.xlabel('Top x% Predictions')
    plt.xticks(x, labels=ratio_top_prediction_list)
    plt.legend(frameon=False, fontsize = 12)

    plt.savefig('Recovery_rate.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return RR_matrix


### Select data based on rules ###

def get_active_learning_trainset(input_file, current_trainset, ratio_top_ave, ratio_top_std, ratio_random):
    """
    get active learning training set
    select the top uncertain compounds from the 'ratio_top_ave' top scoring compounds
    :param input_file: str, path of the input file
    :param current_trainset: str, input file name for current training set
    :param ratio_top_ave: float, the ratio of the top scoring compounds to be selected
    :param ratio_top_std: float, the ratio of the top uncertain compounds to be selected
    :param ratio_random: flaot, the ratio of the random compounds to be selected
    """
    # read files
    df_prediction = pd.read_csv(input_file, index_col = 0)
    num = df_prediction.shape[0]
    assert ratio_top_ave >= ratio_top_std, 'Ratio of top scoring compounds must be larger than number of top uncertain compounds'
    
    # output file
    output_file = os.path.splitext(os.path.abspath(input_file))[0]
    
    # get 'ratio_top_ave' top scoring compounds
    num_top_ave = int(np.around(num * ratio_top_ave))
    print('Number of top scoring compounds:', num_top_ave)
    df_prediction.sort_values(by = ['Ave_Score'], ascending = True, inplace = True)
    df_top_ave = pd.DataFrame(df_prediction.iloc[:num_top_ave])
    
    # get 'ratio_top_std' top uncertain compounds
    num_top_std = int(np.around(num * ratio_top_std))
    print('Number of most uncertain compounds:', num_top_std)
    df_top_ave.sort_values(by = ['Uncertainty'], ascending = False, inplace = True)
    df_top_std = pd.DataFrame(df_top_ave.iloc[:num_top_std])

    # get 'ratio_random' random compounds
    num_random = int(np.round(num * ratio_random))
    print('Number of random compounds:', num_random)
    df_random = df_top_std.sample(n = num_random)
    # df_random.sort_values(by=['Ave_Score'], ascending=True, inplace=True)

    # add Label
    df_random['Label'] = df_random['True_Label']
    
    # write to file
    df_random = df_random.reset_index(drop=True)
    print('Number of rows in new selected file:', df_random.shape[0])
    df_random.to_csv(output_file + '_newSelected.csv')

    # combination with old training set
    df_newSelected = pd.DataFrame(df_random, columns = ['ID', 'Cleaned_SMILES', 'Label'])
    df_currentTrainset = pd.read_csv(current_trainset)
    df_currentTrainset = pd.DataFrame(df_currentTrainset, columns = ['ID', 'Cleaned_SMILES', 'Label'])
    df_newTrainset = pd.concat([df_newSelected, df_currentTrainset], ignore_index=True, sort=False)

    # write to file
    df_newTrainset = df_newTrainset.reset_index(drop=True)
    print('Number of rows in new training set:', df_newTrainset.shape[0])
    df_newTrainset.to_csv(output_file + '_newTrainset.csv')



if __name__ == '__main__':
    
    ### Combine results ###
    srcDir = 'tests/prediction'
    input_file_target = 'tests/target.csv'
    output_file = 'combination'
    combine_multiple_expts(srcDir, input_file_target, output_file)

    ### Recovery rate ###
    prediction_file = 'tests/combination_116.csv'
    target_file = 'tests/target.csv'
    ratio_top_prediction_list = [0.02, 0.04, 0.05, 0.10]
    ratio_top_target_list = [0.02, 0.04, 0.05, 0.10]
    prediction_column_name_list = ['expt0', 'expt1']
    RR_matrix = get_recovery_rate(prediction_file, target_file, ratio_top_prediction_list, ratio_top_target_list, prediction_column_name_list)

    # write recovery rate to .csv output_file
    df = pd.DataFrame(dict(zip(prediction_column_name_list, RR_matrix)))
    df.to_csv('Recovery_rate.csv')

    ### Select data based on rules ###
    input_file = 'tests/test_get_trainset.csv'
    current_trainset = 'tests/target.csv'
    ratio_top_ave = 0.1
    ratio_top_std = 0.05
    ratio_random = 0.01

    get_active_learning_trainset(input_file, current_trainset, ratio_top_ave, ratio_top_std, ratio_random)

