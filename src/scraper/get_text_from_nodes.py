from arguments import create_parser
import pandas as pd
import re
import os
import zipfile
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import gzip


from api import AnnotateHtml, AnnotateHtmlApi

def get_text_spans_from_nodes(text_nodes_df, pred_nodes_df):
    text_pred_nodes_df = pd.merge(pred_nodes_df, text_nodes_df, how='left', on=['Url', 'TextNodeId'])
    return text_pred_nodes_df

def _get_base_filename_by_id(cw22id, cw22root_path, file_type='html'):
        html_path = cw22root_path + os.sep + file_type
        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        language = id_parts[1][:2]
        segment = id_parts[1][:4]
        directory = id_parts[1]
        base_path = html_path + os.sep + language + os.sep + segment + os.sep + directory + os.sep
        base_filename = base_path + id_parts[1] + '-' + id_parts[2]
        return base_filename

def _get_html_from_warc(cw22id, cw22root_path):
        cw22id = cw22id
        cw22root_path = cw22root_path
        base_filename = _get_base_filename_by_id(cw22id, cw22root_path)

        warc_path = base_filename + '.warc.gz'
        offset_path = base_filename + '.warc.offset'

        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        #Get html from warc using offset
        offset_length = len('{:010d}\n'.format(0, 0))
        with open (warc_path,'rb') as f_warc:
            with open (offset_path, 'r') as f_offset:
                f_offset.seek(int(doc) * int(offset_length))
                start_bytes = int (f_offset.read (offset_length).strip())
                end_bytes =   int (f_offset.read (offset_length).strip())
                f_warc.seek(start_bytes)
                record = f_warc.read(end_bytes - start_bytes)
                record = gzip.decompress(record).decode('utf-8')

                #Remove the WARC header to get the htmlStr
                warc_header = ''
                for line in record.splitlines():
                    warc_header += line
                    warc_header += '\r\n'
                    if len(line.strip()) == 0:
                        break
                record = record[len(warc_header):]

                return record
            
def process_file(vdom_path, cw22root_path, filename):
        with zipfile.ZipFile(vdom_path, 'r') as z:
            with z.open(filename) as f:
                data = f.read()
                cw22id = filename[:-4]
                html_string = _get_html_from_warc(cw22id, cw22root_path)

                annotate_html = AnnotateHtml()
                annotate_html.ParseFromString(data)
                api = AnnotateHtmlApi(annotate_html, html_string=html_string)
                all_nodes = api.all_nodes
                url = api.url

                text_nodes = []
                for node in all_nodes.values():
                    if node.is_textnode:
                        text = node.html_node.text.strip('\r\n\t\xa0 ')
                        if len(text) > 0:
                            text_nodes.append({'Url': url, 'TextNodeId': node.nodeid, 'Text': text})

                return text_nodes

def process(vdom_path, cw22root_path):
        text_nodes_df = pd.DataFrame(columns=['Url', 'TextNodeId', 'Text'])
        results_list = []

        with zipfile.ZipFile(vdom_path, 'r') as z:
            filenames = z.namelist()

            with Pool() as pool:
                process_func = partial(process_file, vdom_path, cw22root_path)
                results_list = list(tqdm(pool.imap(process_func, filenames), total=len(filenames)))

        text_nodes_df = pd.concat([pd.DataFrame(result) for result in results_list], ignore_index=True)
        return text_nodes_df

def read_prediction_file(input_path):
    try:
        input_df = pd.read_csv(input_path, delimiter='\t', index_col=False, names=['Url', 'TextNodeId', 'Task'])
        return input_df
    except Exception as e:
        error = str(e)
        raise ValueError('User Error: Failed to read prediction file. Make sure your file is a valid TSV with five columns: Url, TextNodeId, Task. Error: ' + error)


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--model_id", type=str, help="model identifier")
    args = parser.parse_args()
    
    cw22root_path = '/data/public/kqa/clueweb22'
    pattern = r'en\d{4}-\d{2}'
    match = re.findall(pattern, args.model_id)
    entry = match[0][:6]
    zip = match[0]+ ".zip"

    vdom_path = f"/data/public/kqa/clueweb22/vdom/en/en00/{entry}/{zip}"
    print(vdom_path)

    pred_nodes_df = read_prediction_file('test_data/inference_' + match[0] + ".tsv")

    text_nodes_df = process(vdom_path=vdom_path, cw22root_path=cw22root_path)
    #text_nodes_df.to_csv("text_file.csv")

    pred_df = get_text_spans_from_nodes(text_nodes_df, pred_nodes_df).dropna().sort_values(['TextNodeId'], ascending=[False])
    pred_df = pred_df.groupby(['Url', 'Task'], as_index=False).agg({'Text': ''.join})
    
    pred_df = pred_df[['Url', 'Text']]
    pred_df = pred_df.rename(columns=lambda x: x.lower())
    
    output = "baseline_eval_case/" + match[0]+".csv"
    pred_df.to_csv(output,index=False)