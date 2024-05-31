from tokenization import TokenizerProcessor
from api import CommonCrawlApi
import warnings
import json
from bs4 import BeautifulSoup
import chardet
import pandas as pd

CSV_COLUMN_NAMES = ['Url', 'TextNodeId', 'Text']
JSON_COLUMN_NAMES = ['TokenId', 'NodeIds', 'Url']


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



def build(url, raw_html):

    generator = FeatureExtractorApplierProcessor()

    text_nodes_data = []
    json_data = []

    try:
        html_content = raw_html.decode('utf-8')
    except UnicodeDecodeError: 
        # try to figure out encoding if not utf-8

        guess = chardet.detect(raw_html)['encoding']
        if not guess or guess == 'UTF-8': return

        try:
            html_content = raw_html.decode(guess)
        except (UnicodeDecodeError, LookupError):
            # still cant figure out encoding, give up
            return

    html_soup = generator.add_node_id(html_content)
    api = CommonCrawlApi(html_soup=html_soup)

    json_data.extend(generator.Apply(url, api))
        
    all_nodes = api.all_nodes
    for node in all_nodes.values():
        if node.is_textnode:
            text = node.html_node.text.strip('\r\n\t\xa0 ')
            if len(text) > 0:
                text_nodes_data.append([url, node.nodeid, text])

        elif node.html_node.name in ["ol", "dl", "table"]:
            text = node.html_node.text.strip('\r\n\t\xa0 ')
            text_nodes_data.append([url, node.nodeid, text])

    text_nodes_df = pd.DataFrame(text_nodes_data, columns=CSV_COLUMN_NAMES)

    return text_nodes_df, json_data
