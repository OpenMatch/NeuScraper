from tqdm import tqdm
from tokenization import TokenizerProcessor
from multiprocessing import Pool
from api import CommonCrawlApi
import warnings
import json
import os
import csv
from warcio import ArchiveIterator
from argparse import ArgumentParser
from bs4 import BeautifulSoup
import chardet
import pycld2 as cld2
import unicodedata



class FeatureExtractorApplierProcessor:
    def __init__(self):
        self.comment = 'This is the constant comment for all rows returned'
        self.chunk_size = 384
        self.max_token_length = 50


    def _chunk_nodes(self, node_texts, node_seq, node_url):
        chunks = []

        for i in range(0, len(node_texts), self.chunk_size):
            start = i
            end = i + self.chunk_size
            chunk = (node_texts[start:end], node_seq[start:end], node_url[start:end])
            chunks.append(chunk)
            start += self.chunk_size
        
        return chunks
    
    def add_node_id(self,html_str):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
            soup = BeautifulSoup(html_str, 'html.parser')
        all_soup_nodes = soup.find_all()

        stack = []
        stack.append(all_soup_nodes[0])
        node_index = 0
        while len(stack) > 0:
            node = stack.pop()
            if "data-dcnode-id" in node.attrs:
                print("found data-dcnode-id in node attribute, skip this node and its descendants")
            else:
                node.attrs["data-dcnode-id"] = node_index
                node_index += 1

                for child in node.children:
                    if node.name == "span" and isinstance(child, str):
                        continue

                    if isinstance(child, str):
                        new_node = soup.new_tag("span")
                        new_node.string = child
                        new_node.attrs["instrument_node"] = None
                        child.replace_with(new_node)
                        stack.append(new_node)
                    else:
                        stack.append(child)

        return soup
    
    def detect_encoding(self, html_content):
        result = chardet.detect(html_content)
        return result['encoding']
    
   
    def Apply(self, url, api):

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
                        node_url.append(url)

                elif node.html_node.name in ["ol", "dl", "table"]: # List and Table element nodes
                    text = node.html_node.text.strip('\r\n\t\xa0 ')
                    node_sequence.append(node_id)
                    node_texts_tokens.append(tokenizer.tokenize_sequence(text))
                    node_url.append(url)

            # Chunk Document
            chunks = self._chunk_nodes(node_texts_tokens, node_sequence, node_url)

            # Output one row per chunk
            for chunk in chunks:
                json_dict = {'TokenId': chunk[0], 'NodeIds': chunk[1], 'Url': chunk[2]}
                json_str = json.dumps(json_dict, separators=(',', ':'))
                
                yield json_str


# Function to process a single file
def process_file(input):
   with open(path + input, 'rb') as warc_file:
        file_prefix = input.split('.')[0]
        output = file_prefix + ".json"
        csv_output = file_prefix + ".csv"

        with open(output_dir + output, 'a', encoding='utf-8') as json_file,open(textnode_dir + csv_output, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_COLUMN_NAMES)
            
            for record in tqdm(ArchiveIterator(warc_file)):
                    if record.rec_type == 'response':
                        url = record.rec_headers.get_header('WARC-Target-URI')

                        raw_content = record.content_stream().read()
                        encoding = generator.detect_encoding(raw_content)

                        if encoding == None:
                            #print("encode error")
                            continue

                        try:
                            html_content = raw_content.encode('utf-8')
                            
                            try:
                                _,_,details = cld2.detect(html_content)
                            except:
                                # cld2 doesn't like control characters
                                # https://github.com/mikemccand/chromium-compact-language-detector/issues/22#issuecomment-435904616
                                html_no_ctrl_chars = ''.join([l for l in html_content if unicodedata.category(l)[0] not in ['C',]])
                                _,_,details = cld2.detect(html_no_ctrl_chars)

                            if details[0][1] != 'en':
                                continue
                            
                            html_soup = generator.add_node_id(html_content)

                            api = CommonCrawlApi(html_soup=html_soup)
                            x = generator.Apply(url, api)
                            for t in x:
                                json_file.write(f"{t}\n")
                                
                        except:
                            #print("write error")
                            continue
                        
                        all_nodes = api.all_nodes
                        for node in all_nodes.values():
                            if node.is_textnode:
                                text = node.html_node.text.strip('\r\n\t\xa0 ')
                                if len(text) > 0:
                                    csv_writer.writerow([url, node.nodeid, text]) 

                            elif node.html_node.name in ["ol", "dl", "table"]:
                                text = node.html_node.text.strip('\r\n\t\xa0 ')
                                if len(text) > 0:
                                    csv_writer.writerow([url, node.nodeid, text])
                                  



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--path', required=True, help="Path to the directory containing the CommonCrawl WARC files")
    args = parser.parse_args()
    
    generator = FeatureExtractorApplierProcessor()

    CSV_COLUMN_NAMES = ['Url', 'TextNodeId', 'Text']

    path = args.path
    data = os.listdir(path)

    output_dir='commoncrawl/encoded/'
    textnode_dir="commoncrawl/textnodes/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(textnode_dir):
        os.makedirs(textnode_dir)

    with Pool(64) as pool:
        pool.map(process_file, data)