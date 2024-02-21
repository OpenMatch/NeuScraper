import zipfile
import gzip
from tqdm import tqdm
from tokenization import TokenizerProcessor
from argparse import ArgumentParser
from multiprocessing import Pool
from api import AnnotateHtml, AnnotateHtmlApi
import json
import os


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
    
   
    def Apply(self, data, html_string):

            # Load AnnotateHtmlApi
            annotate_html = AnnotateHtml()
            annotate_html.ParseFromString(data)

            api = AnnotateHtmlApi(annotate_html, html_string=html_string)


            # Load tokenizer and global features
            tokenizer = TokenizerProcessor(self.max_token_length)            

            # Build node sequence (text nodes + list/table element nodes)
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

                elif node.html_node.name in ["ol", "dl", "table"]: # List and Table element nodes
                    text = node.html_node.text.strip('\r\n\t\xa0 ')
                    node_sequence.append(node_id)
                    node_texts_tokens.append(tokenizer.tokenize_sequence(text))
                    node_url.append(api.url)

            # Compute Labels
            node_to_annotation = self._get_annotation_labels(api)
            labels = self._compute_labels(node_sequence, node_to_annotation)

            # Chunk Document
            chunks = self._chunk_nodes(node_texts_tokens, labels, node_sequence, node_url)

            # Output one row per chunk
            for chunk in chunks:
                json_dict = {'Labels': chunk[1], 'TokenId': chunk[0], 'NodeIds': chunk[2], 'Url': chunk[3]}
                json_str = json.dumps(json_dict, separators=(',', ':'))
                
                yield json_str


import os
import re
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from functools import partial

# Function to process a single file
def process_file(cw22root_path, entry, zip):
    vdom_path = cw22root_path + f"/vdom/en/en00/{entry}/{zip}"
    name = re.sub(r'\.zip$', '', zip)
    output_file = f"data/train/{name}.json"
    cut_name = name + ".json"
    if cut_name not in test_dataset:
        with open(output_file, 'a', encoding='utf-8') as json_file:
            with zipfile.ZipFile(vdom_path, 'r') as z:
                for filename in tqdm(z.namelist()):
                    with z.open(filename) as f:
                        data = f.read()
                        cw22id = filename[:-4]
                        html_string = generator._get_html_from_warc(cw22id, cw22root_path)
                        x = generator.Apply(data, html_string)
                        for t in x:
                            json_file.write(f"{t}\n")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    
    generator = FeatureExtractorApplierProcessor()

    cw22root_path = args.path
    root = cw22root_path + "/vdom/en/en00/"
    entries = os.listdir(root)
    test_dataset = ["en0001-01.zip"]

    with Pool() as pool:
        process_file_partial = partial(process_file, cw22root_path)
        pool.starmap(process_file_partial, [(entry, zip) for entry in entries for zip in os.listdir(root + entry) if zip.endswith(".zip")])