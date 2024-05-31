# Copyright (c) 2023 OpenMatch
# Author: Zhipeng Xu
# All rights reserved.

import os
import gzip
import json
import zipfile
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
from tokenization import TokenizerProcessor
from api import AnnotateHtml, AnnotateHtmlApi


class FeatureExtractorApplierProcessor:
    def __init__(self):
        self.comment = 'This is the constant comment for all rows returned'
        self.chunk_size = 384
        self.max_token_length = 50


    def _get_html_from_warc(self, cw22id, cw22root_path):
        cw22id = cw22id
        cw22root_path = cw22root_path
        base_filename = self._get_base_filename_by_id(cw22id, cw22root_path)

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

    def _get_base_filename_by_id(self, cw22id, cw22root_path, file_type='html'):
        html_path = cw22root_path + os.sep + file_type
        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        language = id_parts[1][:2]
        segment = id_parts[1][:4]
        directory = id_parts[1]
        base_path = html_path + os.sep + language + os.sep + segment + os.sep + directory + os.sep
        base_filename = base_path + id_parts[1] + '-' + id_parts[2]
        return base_filename

    def _get_annotation_labels(self, api):
        annotation_to_node = api.annotation_to_nodeids
        node_to_annotation = {}
        for ann, node_list in annotation_to_node.items():
            for node in node_list:
                text_nodes = api.get_text_nodeids(node)
                for text_node in text_nodes:
                    if text_node not in node_to_annotation:
                        node_to_annotation[text_node] = []
                    node_to_annotation[text_node].append(ann)
        return node_to_annotation

    def _compute_labels(self, node_sequence, node2ann):
        labels_seq = []
        for node in node_sequence:
            labels = [0] * 6 # Primary + 5 Annotations
            if node in node2ann:
                annotations = node2ann[node]
                for ann in annotations:
                    labels[ann - 1] = 1
            labels_seq.append(labels)
        return labels_seq

    def _chunk_nodes(self, node_texts, labels, node_seq, node_url):
        chunks = []

        for i in range(0, len(node_texts), self.chunk_size):
            start = i
            end = i + self.chunk_size
            chunk = (node_texts[start:end], labels[start:end], node_seq[start:end], node_url[start:end])
            chunks.append(chunk)
            start += self.chunk_size
        
        return chunks


    def Apply(self, api):

        tokenizer = TokenizerProcessor(self.max_token_length)            

        node_sequence = []
        node_texts_tokens = []
        node_url = []
        for node_id, node in api.all_nodes.items():
            if node.is_textnode:
                text = node.html_node.text.strip('\r\n\t\xa0 ') 
                if len(text) > 0:
                    node_sequence.append(node_id)
                    node_texts_tokens.append(tokenizer.tokenize_sequence(text))
                    node_url.append(api.url)

            elif node.html_node.name in ["ol", "dl", "table"]:
                text = node.html_node.text.strip('\r\n\t\xa0 ')
                node_sequence.append(node_id)
                node_texts_tokens.append(tokenizer.tokenize_sequence(text))
                node_url.append(api.url)

        node_to_annotation = self._get_annotation_labels(api)
        labels = self._compute_labels(node_sequence, node_to_annotation)

        chunks = self._chunk_nodes(node_texts_tokens, labels, node_sequence, node_url)

        for chunk in chunks:
            json_dict = {'Labels': chunk[1], 'TokenId': chunk[0], 'NodeIds': chunk[2], 'Url': chunk[3]}
            json_str = json.dumps(json_dict, separators=(',', ':'))
                
            yield json_str


def process_file(filename):
    with zipfile.ZipFile(vdom_path, 'r') as z:
        with z.open(filename) as f:
            data = f.read()
            cw22id = filename[:-4]

            annotate_html = AnnotateHtml()
            annotate_html.ParseFromString(data)
            
            html_string = generator._get_html_from_warc(cw22id, cw22root_path)
            api = AnnotateHtmlApi(annotate_html, html_string=html_string)

            x = generator.Apply(api)

            url = api.url
            all_nodes = api.all_nodes

            rows_features = []
            for t in x:
                rows_features.append(t)

            rows_gt = []
            rows_text = []
            for node in all_nodes.values():
                if node.nodeid in api.annotation_to_nodeids[1]:
                    tag = True
                else:
                    tag = False

                if node.is_textnode:
                    text = node.html_node.text.strip('\r\n\t\xa0 ')
                    if len(text) > 0:
                        rows_gt.append({'Url': url, 'TextNodeId': node.nodeid, 'Text': text, 'JudgmentIsPrimary': tag})
                        rows_text.append({'Url': url, 'TextNodeId': node.nodeid, 'Text': text})

                elif node.html_node.name in ["ol", "dl", "table"]:
                    text = node.html_node.text.strip('\r\n\t\xa0 ')
                    rows_gt.append({'Url': url, 'TextNodeId': node.nodeid, 'Text': text, 'JudgmentIsPrimary': tag})
                    rows_text.append({'Url': url, 'TextNodeId': node.nodeid, 'Text': text})
                        
    return rows_gt, rows_text, rows_features


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()

    if not os.path.exists('data/test/'):
        os.makedirs('data/test/')

    generator = FeatureExtractorApplierProcessor()

    cw22root_path = args.path
    vdom_path = cw22root_path + "/vdom/en/en00/en0001/en0001-01.zip"

    with zipfile.ZipFile(vdom_path, 'r') as z:
        filenames = z.namelist()
    
    with Pool() as pool:
        results = pool.map(process_file, tqdm(filenames))

    rows_gt_all = [row for result in results for row in result[0]]
    rows_text_all = [row for result in results for row in result[1]]
    row_features_all = [row for result in results for row in result[2]]

    gt_nodes_df = pd.DataFrame(rows_gt_all)
    text_nodes_df = pd.DataFrame(rows_text_all)

    gt_nodes_df.to_csv('data/test/GoldLabels.csv',lineterminator='\n', encoding='utf-8', index=False)
    text_nodes_df.to_csv('data/test/TextNodes.csv',lineterminator='\n', encoding='utf-8', index=False)

    output_file = "data/test/TestNodes.json"
    with open(output_file, 'a', encoding='utf-8') as json_file:
        for feature in row_features_all:
            json_file.write(f"{feature}\n")