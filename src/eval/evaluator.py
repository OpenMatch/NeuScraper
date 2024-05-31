# Copyright (c) 2023 OpenMatch
# Author: Zhipeng Xu
# All rights reserved.

import pandas as pd

def read_prediction_file(input_path):
    try:
        input_df = pd.read_csv(input_path, delimiter='\t', index_col=False, names=['Url', 'TextNodeId', 'Task'])
        return input_df
    except Exception as e:
        error = str(e)
        raise ValueError('User Error: Failed to read prediction file. Make sure your file is a valid TSV with five columns: Url, TextNodeId, Task. Error: ' + error)

def read_primary_ground_truth_file(gt_path):
    try:
        input_df = pd.read_csv(gt_path, delimiter='\t', header=0, quoting=3, names=['Url', 'TextNodeId', 'Text', 'JudgmentIsPrimary'],dtype={'Url': str, 'TextNodeId': int, 'Text': str, 'JudgmentIsPrimary': bool})
        return input_df
    except Exception as e:
        error = str(e)
        raise SystemError('Internal Error: Failed to read primary ground truth file.')
        
def get_text_spans_from_nodes(text_nodes_df, pred_nodes_df):
    text_pred_nodes_df = pd.merge(pred_nodes_df, text_nodes_df, how='left', on=['Url', 'TextNodeId'])
    return text_pred_nodes_df

def compute_metrics(tp, tn, fp, fn):
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    fscore = (2*tp)/(2*tp + fp + fn)

    return precision, recall, accuracy, fscore

def compute_primary_task_metrics_from_text(pred_text_df, positive_gt, negative_gt):
    tp, tn, fp, fn = 0., 0., 0., 0.
    for _, row in pred_text_df.iterrows():
        page_tp, page_tn, page_fp, page_fn = evaluate_predictions(row['Text'], positive_gt[row['Url']], negative_gt[row['Url']])
        tp += page_tp
        tn += page_tn
        fp += page_fp
        fn += page_fn
    
    return compute_metrics(tp, tn, fp, fn)


def get_primary_ground_truth_text_dicts(ground_truth_text_df):
    primary_gt = {}
    secondary_gt = {}
    for _, row in ground_truth_text_df.iterrows():
        if row['Url'] not in primary_gt:
            primary_gt[row['Url']] = []
            secondary_gt[row['Url']] = []

        if row['JudgmentIsPrimary']:
            primary_gt[row['Url']].append(row['Text'])
        else:
            secondary_gt[row['Url']].append(row['Text'])
    return primary_gt, secondary_gt

def evaluate_predictions(pred_text, positives, negatives):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    if pred_text is None or type(pred_text) is not str:
        false_negatives += len(positives)
        true_negatives += len(negatives)
        return true_positives, true_negatives, false_positives, false_negatives


    for positive in positives:
        if positive is not None and type(positive) is str:
            positive = positive.replace('#N#', '\n').replace('#TAB#', '\t').replace('#R#', '\r')

            if positive in pred_text:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            false_negatives += len(positives)
    
    for negative in negatives:
        if negative is not None and type(negative) is str:
            negative = negative.replace('#N#', '\n').replace('#TAB#', '\t').replace('#R#', '\r')
            if negative in pred_text:
                false_positives += 1
            else:
                true_negatives += 1
        else:
            true_negatives += len(negatives)
    
    return true_positives, true_negatives, false_positives, false_negatives