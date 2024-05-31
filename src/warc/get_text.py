import pandas as pd
import os
import re
from tqdm import tqdm

def get_text_spans_from_nodes(text_nodes_df, pred_nodes_df):
    text_pred_nodes_df = pd.merge(pred_nodes_df, text_nodes_df, how='left', on=['Url', 'TextNodeId'])
    return text_pred_nodes_df

def read_prediction_file(input_path):
    try:
        input_df = pd.read_csv(input_path, delimiter='\t', index_col=False, names=['Url', 'TextNodeId', 'Task'])
        return input_df
    except Exception as e:
        error = str(e)
        raise ValueError('User Error: Failed to read prediction file. Make sure your file is a valid TSV with five columns: Url, TextNodeId, Task. Error: ' + error)


if __name__ == "__main__":


    if not os.path.exists('commoncrawl/text'):
        os.makedirs('commoncrawl/text')

    for file in tqdm(os.listdir("commoncrawl/temp")):
        try:
            pattern = r'([^\\\/]+)(?=\.tsv)'
            match = re.search(pattern, file)

            name = match.group(0)

            pred_nodes_df = read_prediction_file('commoncrawl/temp/'+ name + ".tsv")

            text_nodes_df = pd.read_csv("commoncrawl/textnodes/" + name + ".csv")
            text_nodes_df.columns = ['Url', 'TextNodeId', 'Text']
            pred_df = get_text_spans_from_nodes(text_nodes_df, pred_nodes_df).dropna().sort_values(['TextNodeId'], ascending=[False])
            pred_df = pred_df.groupby(['Url', 'Task'], as_index=False).agg({'Text': ''.join})
            
            pred_df = pred_df[['Url', 'Text']]
            pred_df = pred_df.rename(columns=lambda x: x.lower())
            
            output = "commoncrawl/text/" + name + ".csv"
            pred_df.to_csv(output,index=False)
        except:
            print("error, skipping file: " + file)
            continue