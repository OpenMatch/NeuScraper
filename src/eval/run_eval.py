# Copyright (c) 2023 OpenMatch
# Author: Zhipeng Xu
# All rights reserved.

import evaluator
import pandas as pd
import numpy as np

def sort(pred_df, text_df):

    pred_df = pred_df.drop(columns=['Task'])
    
    url_df = pd.DataFrame({'Url': text_df['Url'].unique()})
    urls_to_add = url_df[~url_df['Url'].isin(pred_df['Url'])]['Url']
    url_order = {url: idx for idx, url in enumerate(url_df['Url'])}

    dropped_rows = pd.DataFrame({
        'Url': urls_to_add,
        'Text': np.nan
    })

    pred_df = pd.concat([pred_df, dropped_rows], ignore_index=True)
    pred_df['SortOrder'] = pred_df['Url'].map(url_order)
    pred_df_sorted = pred_df.sort_values(by='SortOrder')
    pred_df_sorted = pred_df_sorted.drop(columns=['SortOrder'])
    pred_df_sorted.reset_index(drop=True, inplace=True)
    
    return pred_df_sorted


if __name__ == '__main__':


    gt_df = pd.read_csv("data/test/GoldLabels.csv",lineterminator='\n').dropna()
    positive_text, negative_text = evaluator.get_primary_ground_truth_text_dicts(gt_df)

    pred_df = evaluator.read_prediction_file("temp/inference_test.tsv")
    pred_df = pred_df[pred_df['Task'] == 'Primary']
    pred_df["TextNodeId"] = pred_df["TextNodeId"].astype(int)

    text_df = pd.read_csv("data/test/TextNodes.csv",lineterminator='\n')
    text_df["TextNodeId"] = text_df["TextNodeId"].astype(int)
    
    pred_df = evaluator.get_text_spans_from_nodes(text_df, pred_df).dropna().sort_values(['TextNodeId'], ascending=[False])
    pred_df = pred_df.groupby(['Url', 'Task'], as_index=False).agg({'Text': ''.join})

    pred_df = sort(pred_df, text_df)
    precision, recall, accuracy, fscore = evaluator.compute_primary_task_metrics_from_text(pred_df, positive_text, negative_text)

    pred_df.to_csv('temp/neuscraper.csv', index=False)
    print("Metrics for NeuScraper: Acc: %f Prec: %f Rec: %f F1: %f" % (accuracy, precision, recall, fscore))
