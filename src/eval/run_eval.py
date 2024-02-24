import evaluator
import pandas as pd
import time

if __name__ == '__main__':


    gt_df = pd.read_csv("data/test/GoldLabels.csv",lineterminator='\n').dropna()
    positive_text, negative_text = evaluator.get_primary_ground_truth_text_dicts(gt_df)

    pred_df = evaluator.read_prediction_file("temp/inference_test.tsv")
    pred_df = pred_df[pred_df['Task'] == 'Primary']
    text_df = pd.read_csv("data/test/TextNodes.csv",lineterminator='\n')

    pred_df = evaluator.get_text_spans_from_nodes(text_df, pred_df).dropna().sort_values(['TextNodeId'], ascending=[False])
    pred_df = pred_df.groupby(['Url', 'Task'], as_index=False).agg({'Text': ''.join})
    precision, recall, accuracy, fscore = evaluator.compute_primary_task_metrics_from_text(pred_df, positive_text, negative_text)


    print("Metrics for NeuScraper: Prec: %f Rec: %f Acc: %f F1: %f" % (precision, recall, accuracy, fscore))
