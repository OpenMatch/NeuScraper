from cgitb import text
from mimetypes import init
from traceback import format_exc
from .AnnotateHtml_pb2 import AnnotateHtml
from bs4 import BeautifulSoup
import re
import string 
import queue
import time
import os
import gzip

REGEX_PATTERN = re.compile(r'([a-z][a-z])(\d+)')

INLINE_ELEMENT = { "a", "abbr", "acronym", "b", "bdo", "big", "br", "cite", "code", "dfn", "em", "i", "img", "input", "kbd", "label", "map", "object", "q", "samp", "script", "small", "span", "strong", "sub", "sup", "textarea", "time", "tt", "var" }

BLOCK_ELEMENT = {"address", "article", "aside", "blockquote", "canvas", "dd", "div", "dl", "dt", 
"fieldset", "figcaption", "caption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
 "h6", "header", "hgroup", "hr", "li", "main", "nav", "noscript", "ol", "output", "p", "pre", 
 "section", "table", "tr", "td", "th", "tbody", "thead", "tfoot", "ul", "video", "h7", "h8", "h9", 
 "applet", "audio", "button", "datalist", "frameset", "legend", "menu", "summary", "textarea", "title"
}

TABLE_ELEMENT = {"caption", "th", "tr", "td"}
LIST_ELEMENT = {"li"}
TABLE_LIST_ELEMENT_ANNOTATION_NAME = {"caption" : "TableCaption", "th" : "TableHeader", "tr" : "TableRow", "td" : "TableCell", "li" : "ListItem"}

class AnnotateHtmlApi:
    class AnnotateHtmlNode:
        def __init__(self, id):
            self.nodeid = id
            self.vdom_feature = None
            self.html_node = None
            self.is_textnode = False
            self.annotations = set()
            self.first_textnode = None
            self.last_textnode = None
            self.children_nodes = []
            self.parent_node = None
            self.prev_node = None
            self.next_node = None
            self.last_descendant_node = None

            # only for text node
            self.prev_textnode = None
            self.next_textnode = None


    def __init__(self, annotate_html, init_nodes=True, html_string=''):
        if not isinstance(annotate_html, AnnotateHtml):
            raise ValueError("Object passed in is not a valid AnnotateHtml Object")
        
        self.annotateHtml = annotate_html
        if len(html_string) > 0:
            self.htmlStr = html_string
            self.soup = BeautifulSoup(self.htmlStr, 'html.parser')
        self.url = annotate_html.url
        self.urlhash = annotate_html.urlhash
        self.cw22id = annotate_html.cw22id
        self.language = annotate_html.language
        self.topic = list(annotate_html.topic)
        self.annotations = annotate_html.annotations

        self.all_nodes = {}
        self.nodeid_to_feature = {}
        self.textnode_id_in_order = []
        self.html_title_node = None

        #initialize node class and relationship
        if init_nodes and len(html_string.strip()) > 0:
            self.init_all_nodes()
            
            self.annotation_to_nodeids = {}
            self.init_annotation_map()

    def get_all_node_features_no_offset(self):
        node_vdom_features = []
        for attr_str in self.annotateHtml.nodeRawFeature:
            vdom_feature = self.parse_node_raw_attribute(attr_str)
            node_vdom_features.append(vdom_feature)

        return node_vdom_features

    def parse_node_raw_attribute(self, attribute_str):
        vdom_feature = self.init_vdom_feature()

        node_id = 0
        regex_result = REGEX_PATTERN.finditer(attribute_str)

        for match in regex_result:
            attr_name = match[1]
            attr_value = int(match[2])
            
            if attr_name == "id": node_id = attr_value 
            elif attr_name ==  "px": vdom_feature.position_x = attr_value 
            elif attr_name ==  "py": vdom_feature.position_y= attr_value 
            elif attr_name ==  "pw": vdom_feature.position_w = attr_value 
            elif attr_name ==  "ph": vdom_feature.position_h = attr_value 
            elif attr_name ==  "ol": vdom_feature.offset_left = attr_value 
            elif attr_name ==  "ot": vdom_feature.offset_top = attr_value 
            elif attr_name ==  "ow": vdom_feature.offset_w = attr_value 
            elif attr_name ==  "oh": vdom_feature.offset_h = attr_value 
            elif attr_name ==  "cl": vdom_feature.client_left = attr_value 
            elif attr_name ==  "ct": vdom_feature.client_top = attr_value 
            elif attr_name ==  "cw": vdom_feature.client_w = attr_value 
            elif attr_name ==  "ch": vdom_feature.client_h = attr_value 
            elif attr_name ==  "ca": vdom_feature.font_color_a = attr_value 
            elif attr_name ==  "cr": vdom_feature.font_color_r = attr_value 
            elif attr_name ==  "cb": vdom_feature.font_color_b = attr_value 
            elif attr_name ==  "cg": vdom_feature.font_color_g = attr_value 
            elif attr_name ==  "fw": vdom_feature.font_weight = attr_value 
            elif attr_name ==  "fs": vdom_feature.font_size = attr_value 
            elif attr_name ==  "fi": vdom_feature.font_italic = attr_value 
            elif attr_name ==  "fd": vdom_feature.font_decoration = attr_value 
            elif attr_name ==  "lt": vdom_feature.list_style = attr_value 
            elif attr_name ==  "di": vdom_feature.display_style = attr_value 
            elif attr_name ==  "cu": vdom_feature.cursor_style = attr_value 
            elif attr_name ==  "lh": vdom_feature.line_height = attr_value 
            elif attr_name ==  "tt": vdom_feature.text_transform = attr_value 
            elif attr_name ==  "op": vdom_feature.opacity = attr_value 
            elif attr_name ==  "bl": vdom_feature.border_style_left = attr_value 
            elif attr_name ==  "bt": vdom_feature.border_style_top = attr_value 
            elif attr_name ==  "br": vdom_feature.border_style_right = attr_value 
            elif attr_name ==  "bb": vdom_feature.border_style_bottom = attr_value 

        if node_id in self.all_nodes:
            self.all_nodes[node_id].vdom_feature = vdom_feature

        vdom_dict = {'node_id': node_id, 'vdom_feature': vdom_feature}
        return vdom_dict


    def init_all_nodes(self):
        all_soup_nodes = self.soup.find_all()

        temp_prev_node = None
        temp_prev_textnode = None
        for htmlnode in all_soup_nodes:
            # all valid and valuable nodes have 'data-dcnode-id' attribute
            if 'data-dcnode-id' in htmlnode.attrs:
                nodeid = int(htmlnode.attrs['data-dcnode-id'])

                annotatehtml_node = self.AnnotateHtmlNode(nodeid)
                annotatehtml_node.html_node = htmlnode

                # all textnode has 'instrument_node' attribute
                if 'instrument_node' in htmlnode.attrs and htmlnode.parent.name != "noscript":
                    annotatehtml_node.is_textnode = True

                    # assign previous textnode as previous textnode
                    annotatehtml_node.prev_textnode = temp_prev_textnode

                    # assign current textnode as next textnode for previous node
                    if temp_prev_textnode != None:
                        self.all_nodes[temp_prev_textnode.nodeid].next_textnode = annotatehtml_node
                    
                    temp_prev_textnode = annotatehtml_node
                    self.textnode_id_in_order.append(nodeid)

                # assign previous node as previous node
                annotatehtml_node.prev_node = temp_prev_node

                # assign current node as next node for previous node
                if temp_prev_node != None:
                    self.all_nodes[temp_prev_node.nodeid].next_node = annotatehtml_node
                
                temp_prev_node = annotatehtml_node

                self.all_nodes[nodeid] = annotatehtml_node

                # add first <title> tag as html title node
                if htmlnode.name == "title" and self.html_title_node == None:
                    self.html_title_node = annotatehtml_node

        # initialize node parent/children, first/next text node relationship 
        self.compute_node_relationship()

        # initialize node features
        for attr_str in self.annotateHtml.nodeRawFeature:
            self.parse_node_raw_attribute(attr_str)

        # check node without vdom feature
        all_node_ids = list(self.all_nodes.keys())
        all_node_ids.sort(reverse=False)
        for nodeid in all_node_ids:
            if self.all_nodes[nodeid].vdom_feature == None:
                if self.all_nodes[nodeid].parent_node != None and self.all_nodes[nodeid].vdom_feature != None:
                    self.all_nodes[nodeid].vdom_feature = self.all_nodes[nodeid].parent_node.vdom_feature
                else:
                    self.all_nodes[nodeid].vdom_feature = self.init_vdom_feature()


    def compute_node_relationship(self):
        all_node_ids = list(self.all_nodes.keys())
        # after sort, children nodes is being processed early than parent node
        # parent node could inherit first / last text node information from children nodes 
        all_node_ids.sort(reverse=True)

        for curr_nodeid in all_node_ids:
            if not self.all_nodes[curr_nodeid].is_textnode:
                curr_htmlnode = self.all_nodes[curr_nodeid].html_node                
                temp_last_textnode = None
                max_descendant_nodeid = -1

                for child in curr_htmlnode.find_all(recursive=False):
                    if 'data-dcnode-id' in child.attrs:
                        child_node_id = int(child.attrs['data-dcnode-id'])

                        # add child node to children node list
                        self.all_nodes[curr_nodeid].children_nodes.append(self.all_nodes[child_node_id])
                        
                        # assign parent node to child
                        self.all_nodes[child_node_id].parent_node = self.all_nodes[curr_nodeid]

                        # if first text node not found, update with child's first text node id
                        if self.all_nodes[curr_nodeid].first_textnode == None and self.all_nodes[child_node_id].first_textnode != None:
                            self.all_nodes[curr_nodeid].first_textnode = self.all_nodes[child_node_id].first_textnode

                        # update last text node id with current node's last text node id
                        if self.all_nodes[child_node_id].last_textnode != None:
                            temp_last_textnode = self.all_nodes[child_node_id].last_textnode
                        
                        child_last_descendant_id = self.all_nodes[child_node_id].last_descendant_node.nodeid if self.all_nodes[child_node_id].last_descendant_node != None else child_node_id
                        max_descendant_nodeid = max(max_descendant_nodeid, child_last_descendant_id)

                if temp_last_textnode != None:
                    self.all_nodes[curr_nodeid].last_textnode = temp_last_textnode
                
                if max_descendant_nodeid != -1:
                    self.all_nodes[curr_nodeid].last_descendant_node  = self.all_nodes[max_descendant_nodeid]
                else:
                    self.all_nodes[curr_nodeid].last_descendant_node = self.all_nodes[curr_nodeid]

            # if it's text node, update first and last as it self
            else:
               self.all_nodes[curr_nodeid].first_textnode = self.all_nodes[curr_nodeid]
               self.all_nodes[curr_nodeid].last_textnode = self.all_nodes[curr_nodeid]
               self.all_nodes[curr_nodeid].last_descendant_node = self.all_nodes[curr_nodeid]


    def init_annotation_map(self):
        for annotation in self.annotateHtml.annotations:
            if annotation.nodeId in self.all_nodes.keys():
                self.all_nodes[annotation.nodeId].annotations.add(annotation.type)

                if annotation.type not in self.annotation_to_nodeids.keys():
                    self.annotation_to_nodeids[annotation.type] = []
                
                self.annotation_to_nodeids[annotation.type].append(annotation.nodeId)
            else:
                #print("warning: annotation key not found: " + str(annotation.nodeId))
                pass

        # fall back to full content when primary is empty
        if AnnotateHtml.AnnotationType.Primary not in self.annotation_to_nodeids:
            self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Primary] = self.textnode_id_in_order
        
        for annotation_type in self.annotation_to_nodeids:
            self.annotation_to_nodeids[annotation_type].sort(reverse=True)        


    # get text node ids for any given node
    def get_text_nodeids(self, nodeid):
        textnodeids = []
        if nodeid not in self.all_nodes:
            return textnodeids

        curr_textnode = self.all_nodes[nodeid].first_textnode

        last_textnode = self.all_nodes[nodeid].last_textnode

        while curr_textnode != None and curr_textnode.nodeid >= last_textnode.nodeid:
            textnodeids.append(curr_textnode.nodeid)
            curr_textnode = curr_textnode.next_textnode
        
        return textnodeids
    

    # return normalized primary text of document
    def get_primary_content(self, add_html_title=True):

        if AnnotateHtml.AnnotationType.Primary in self.annotation_to_nodeids.keys():
            primary_nodeids = self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Primary]
    
            if len(primary_nodeids) == 0:
                return ("", {})

            return self.get_content_for_nodes(primary_nodeids, add_html_title=add_html_title)
        # change to fallback
        else:
            return ("", {})


     # return normalized primary text of document with annotation start end index offset
    def get_primary_content_with_annotation_offset(self, get_binary_text, add_html_title=True):

        if AnnotateHtml.AnnotationType.Primary in self.annotation_to_nodeids.keys():
            primary_nodeids = self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Primary]
    
            if len(primary_nodeids) == 0:
                if get_binary_text:
                    empty_binary = " ".encode("utf-8")
                    return (empty_binary, {})
                else:
                    return (" ", {})

            primary_content_with_offset =  self.get_content_for_nodes(primary_nodeids, get_binary=get_binary_text, add_html_title=add_html_title)
            textnode_offset = primary_content_with_offset[1]

            # add annotation offsets
            annotation_offset = {}

            if add_html_title and self.html_title_node != None and self.html_title_node.nodeid in textnode_offset:
                annotation_offset["HtmlTitle"] = {self.html_title_node.nodeid : textnode_offset[self.html_title_node.nodeid]}

            if AnnotateHtml.AnnotationType.Title in self.annotation_to_nodeids:
                title_nodeids = self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Title]                
                title_offsets = self.get_title_primary_content_offset(textnode_offset, title_nodeids)
                annotation_offset["Title"] = title_offsets

            if AnnotateHtml.AnnotationType.Heading in self.annotation_to_nodeids:
                heading_nodeids = self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Heading]
                heading_offsets = self.get_node_content_offset(textnode_offset, heading_nodeids)
                annotation_offset["Heading"] = heading_offsets
            
            if AnnotateHtml.AnnotationType.Paragraph in self.annotation_to_nodeids:
                paragraph_nodeids = self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Paragraph]
                paragraph_offsets = self.get_node_content_offset(textnode_offset, paragraph_nodeids)
                annotation_offset["Paragraph"] = paragraph_offsets
            
            # Table annotation will add table caption, table header, table row, table cell annotation
            if AnnotateHtml.AnnotationType.Table in self.annotation_to_nodeids:
                # find table related element for each table
                for table_nodeid in self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Table]:                    
                    table_related_annotation = self.find_node_in_children(self.all_nodes[table_nodeid], TABLE_ELEMENT)                    

                    for tagname in TABLE_ELEMENT:
                        if len(table_related_annotation[tagname]) > 0:                            
                            tag_offsets = self.get_node_content_offset(textnode_offset, table_related_annotation[tagname])
                            # remove nested table row or table cell
                            skip_node_ids = set()
                            if tagname == "tr" or tagname == "td":
                                skip_node_ids = self.remove_nested_node(list(tag_offsets.keys()))

                            annotation_name = TABLE_LIST_ELEMENT_ANNOTATION_NAME[tagname]                                
                            
                            if annotation_name not in annotation_offset:
                                annotation_offset[annotation_name] = {}

                            for nodeid in tag_offsets:
                                if nodeid not in skip_node_ids:
                                    annotation_offset[annotation_name][nodeid] = tag_offsets[nodeid]
                
                table_offsets = self.get_node_content_offset(textnode_offset, self.annotation_to_nodeids[AnnotateHtml.AnnotationType.Table])
                annotation_offset["Table"] = table_offsets
            
            # List annotation will add list item annotation
            if AnnotateHtml.AnnotationType.List in self.annotation_to_nodeids:
                # find table related element for each table
                for list_nodeid in self.annotation_to_nodeids[AnnotateHtml.AnnotationType.List]:                    
                    list_related_annotation = self.find_node_in_children(self.all_nodes[list_nodeid], LIST_ELEMENT)                    

                    for tagname in LIST_ELEMENT:
                        if len(list_related_annotation[tagname]) > 0:     
                            tag_offsets = self.get_node_content_offset(textnode_offset, list_related_annotation[tagname])
                            # remove nested list item
                            skip_node_ids = set()
                            if tagname == "li":
                                skip_node_ids = self.remove_nested_node(list(tag_offsets.keys()))

                            annotation_name = TABLE_LIST_ELEMENT_ANNOTATION_NAME[tagname]
                            
                            if annotation_name not in annotation_offset:
                                annotation_offset[annotation_name] = {}

                            for nodeid in tag_offsets:
                                if nodeid not in skip_node_ids:
                                    annotation_offset[annotation_name][nodeid] = tag_offsets[nodeid]
                    
                list_offsets = self.get_node_content_offset(textnode_offset, self.annotation_to_nodeids[AnnotateHtml.AnnotationType.List])
                annotation_offset["List"] = list_offsets

            # add invisible annotation 
            invisible_start = -1
            invisible_end = -1
            # invisible could continue for multiple nodes, we only use first invisible node as key
            invisible_start_nodeid = -1
            for textnodeid in textnode_offset:
                textnode = self.all_nodes[textnodeid]
                if self.html_title_node != None and textnodeid == self.html_title_node.nodeid:
                    continue
                
                if not self.is_node_visible(textnode):
                    if invisible_start == -1:
                        invisible_start = textnode_offset[textnodeid][0]
                        invisible_start_nodeid = textnodeid
                    
                    invisible_end = textnode_offset[textnodeid][0]
                    
                elif invisible_start != -1 and invisible_end != -1:
                    if "InvisibleText" not in annotation_offset:
                        annotation_offset["InvisibleText"] = {}

                    annotation_offset["InvisibleText"][invisible_start_nodeid] = (invisible_start, invisible_end)
                    invisible_start = -1
                    invisible_end = -1
                    invisible_start_nodeid = -1
            
            if invisible_start != -1 and invisible_end != -1:
                if "InvisibleText" not in annotation_offset:
                    annotation_offset["InvisibleText"] = {}

                annotation_offset["InvisibleText"][invisible_start_nodeid] = (invisible_start, invisible_end)

            return (primary_content_with_offset[0], annotation_offset)

        else:
            return ("", {})

    # function to get title annotation offset in given nodeid to offset map
    # separate from other annotation because title annotation is on text node
    def get_title_primary_content_offset(self, nodeid_to_offset, title_nodeids):
        # offset list    
        offsets = {}

        start_index = -1
        end_index  = -1

        for title_nodeid in title_nodeids:
            if title_nodeid not in nodeid_to_offset:
                continue

            parent_nodeid = self.all_nodes[title_nodeid].parent_node.nodeid

            start_index = nodeid_to_offset[title_nodeid][0]
            end_index = nodeid_to_offset[title_nodeid][1]

            if start_index == -1 or end_index == -1:
                continue

            if parent_nodeid in offsets:
                offsets[parent_nodeid] = (offsets[parent_nodeid][0], end_index)
            else:
                offsets[parent_nodeid] = (start_index, end_index)

        return offsets

    #   get annotation nodes offset in given nodeid to offset map other than title
    def get_node_content_offset(self, nodeid_to_offset, nodeids):  
        offsets = {}

        for nodeid in nodeids:
            curr_textnode = self.all_nodes[nodeid].first_textnode
            last_textnode = self.all_nodes[nodeid].last_textnode
            start_index = -1
            end_index  = -1

            while curr_textnode != None and curr_textnode.nodeid >= last_textnode.nodeid:

                if curr_textnode.nodeid in nodeid_to_offset:

                    if start_index == -1:
                        start_index = nodeid_to_offset[curr_textnode.nodeid][0]
                    
                    end_index = nodeid_to_offset[curr_textnode.nodeid][1]

                curr_textnode = curr_textnode.next_textnode
                
            if start_index != -1 and end_index != -1:
                offsets[nodeid] = (start_index, end_index)

        return offsets


    # remove nested node id, keep only highest level node, return list of nodeids that should be removed 
    def remove_nested_node(self, nodeid_list):
        # small node id first, make sure parent node get processed early than children node
        sorted_nodeid = sorted(nodeid_list, reverse=False)
        included_nodeid = set()
        should_skip_nodeid = set()

        for nodeid in sorted_nodeid:
            if nodeid in included_nodeid:
                should_skip_nodeid.add(nodeid)
            
            included_nodeid.update(range(nodeid, self.all_nodes[nodeid].last_descendant_node.nodeid))
        
        return should_skip_nodeid


    # return normalized text given list of node ids
    def get_content_for_nodes(self, nodeids, get_binary, add_html_title=True, trim_space=True, default_text_separator="\n"):
        nodeid_to_start_offset = {}
        nodeid_to_end_offset = {}
        nodeid_to_offset = {}
        output_text = ""

        if get_binary:
            output_text = output_text.encode("utf-8")

        prev_textnode = None
        curr_textnode = None

        if add_html_title and self.html_title_node != None:
            title_text = self.html_title_node.html_node.text.strip('\r\n\t\xa0 ') if trim_space else self.html_title_node.html_node.text

            if get_binary:
                title_text= title_text.encode("utf-8")

            output_text = output_text + title_text
            if len(output_text) != 0:
                nodeid_to_offset[self.html_title_node.nodeid] = (0, len(output_text))

        for nodeid in nodeids:
            if nodeid in self.all_nodes.keys():
                
                annotatehtml_node = self.all_nodes[nodeid]
                
                # get text content
                node_text = annotatehtml_node.html_node.text.strip('\r\n\t\xa0 ') if trim_space else annotatehtml_node.html_node.text

                if node_text == "":
                    continue
                
                if get_binary:
                    node_text = node_text.encode("utf-8")

                curr_textnode = annotatehtml_node.first_textnode
                text_separator = default_text_separator

                # if current node doesn't have first text node
                if curr_textnode == None :
                    text_separator = "\n"
                else:
                    text_separator = self.get_text_separator(prev_textnode, curr_textnode, default_text_separator)

                # remove separator if this is beginning of text
                if len(output_text) == 0:
                    text_separator = ""
                
                if get_binary:
                    text_separator = text_separator.encode("utf-8")

                # save start index before append text
                nodeid_to_start_offset= len(output_text) + len(text_separator)

                output_text = output_text + text_separator + node_text

                # save end index after append text
                nodeid_to_end_offset = len(output_text)

                nodeid_to_offset[nodeid] = (nodeid_to_start_offset, nodeid_to_end_offset)
                # update previous text node
                prev_textnode = curr_textnode

            else:
                print("warning: nodeid not found in all_nodes: " + str(nodeid))

        return (output_text, nodeid_to_offset)


    def get_text_separator(self, prev_textnode, curr_textnode, default_separator = "\n"):        
        separator = default_separator

        if prev_textnode == None:
            return separator

        prev_textnode_visible = self.is_node_visible(prev_textnode)
        cur_textnode_visible = self.is_node_visible(curr_textnode)
        
        # \n separator between visible boundary
        if prev_textnode_visible != cur_textnode_visible:
            separator = "\n"

        # check if there is <br> tag between two text nodes
        elif self.is_br_between_textnodes(prev_textnode, curr_textnode):
            separator = "\n"

        # check if prev and current textnode not in same block element
        elif not self.is_under_same_block(prev_textnode, curr_textnode):
            separator = "\n"
        
        # check if prev and current textnode has space in between
        elif self.is_space_in_between(prev_textnode, curr_textnode):
            separator = " "

        # inline elements under same block no separator
        elif self.is_node_inlineElement(prev_textnode) and self.is_node_inlineElement(curr_textnode) and not self.has_margin_or_padding_between(prev_textnode, curr_textnode):
            separator = ""

        elif self.starts_with_punctuatin_with_space( curr_textnode) or self.ends_with_space_and_punctuation(prev_textnode):
            separator = ""
        else:
            separator = " "

        return separator


    def is_node_visible(self, node):
        # title node is not visible
        if node.html_node == "title":
            return False
        
        node_feature = node.vdom_feature

        if node_feature.position_w <= 2 or node_feature.position_h <= 2 or node_feature.opacity == 0:
            return False

        return True
    

    def is_br_between_textnodes(self, prev_textnode, curr_textnode):       
        
        temp_prev_node = curr_textnode.prev_node

        while temp_prev_node != None:
            # check if it's br
            if temp_prev_node.html_node.name == "br":
                return True

            # if meet prev text node, break while loop 
            if temp_prev_node == prev_textnode:
                break
                
            temp_prev_node = temp_prev_node.prev_node

        return False


    def is_under_same_block(self, prev_textnode, curr_textnode):
        prev_innermost_block_node = self.get_innermost_block_element(prev_textnode)
        curr_innermost_block_node = self.get_innermost_block_element(curr_textnode)

        if (prev_innermost_block_node != None and curr_innermost_block_node != None 
        and prev_innermost_block_node.nodeid == curr_innermost_block_node.nodeid):
            return True            
    
        # treat as same block if current block is inline style
        # elif self.is_node_inlineElement(curr_textnode):
        #     return True

        return False

        
    def get_innermost_block_element(self, node):
        while node != None and node.html_node.name not in BLOCK_ELEMENT:
            node = node.parent_node
        
        return node

    def is_node_blockElement(self, node):
        if node.is_textnode:
            if node.parent_node == None:
                print("Warning: node doesn't have parent node, id: " + str(node.nodeid))
                return True
            else:
                node = node.parent_node

        parent_tag_name = node.html_node.name
        return parent_tag_name in BLOCK_ELEMENT

    def is_node_inlineElement(self, node):
        if node.is_textnode:
            if node.parent_node == None:
                print("Warning: node doesn't have parent node, id: " + str(node.nodeid))
                return True
            else:
                node = node.parent_node

        if "style" in node.html_node.attrs and "display:inline" in node.html_node.attrs["style"]:
            return True

        parent_tag_name = node.html_node.name
        return parent_tag_name in INLINE_ELEMENT


    def is_space_in_between(self, prev_textnode, curr_textnode):
        prev_text = prev_textnode.html_node.text
        curr_text = curr_textnode.html_node.text

        # s1: previous text ends with space or tab
        if len(prev_text) > 0 and (prev_text[-1] == " " or prev_text[-1] == "\t"):
            return True
        # s2: current text starts with space or tab   
        if len(curr_text) > 0 and (curr_text[0] == " " or curr_text[0] == "\t"):
            return True
        
        return False


    def has_margin_or_padding_between(self, prev_textnode, curr_textnode):
        if prev_textnode == None:
            return True
        
        prev_feature = prev_textnode.vdom_feature
        curr_feature = curr_textnode.vdom_feature

        # visual space 
        if prev_feature.position_x + prev_feature.position_w + 5 <= curr_feature.position_x:
            return True
        
        # offset
        if curr_feature.offset_left >= 5 or curr_feature.client_left >= 5:
            return True


    def starts_with_punctuatin_with_space(self, curr_textnode):
        text = curr_textnode.html_node.text

        if text.isspace() or len(text) < 1 or not text[0] in string.punctuation:
            return False
        
        # there is one letter and it is punctuation symbol
        if len(text) == 1:
            return True
        
        # if there are more than one letter, then first letter must be punctuation symbol and second letter is blank space
        if text[1] == " ":
            return True
        
        return False


    def ends_with_space_and_punctuation(self, curr_textnode):
        text = curr_textnode.html_node.text

        if text.isspace() or len(text) < 1 or not text[-1] in string.punctuation:
            return False
        
        # there is one letter and it is punctuation symbol
        if len(text) == 1:
            return True
        
        # if there are more than one letter, then last letter must be punctuation symbol and second last letter is blank space
        if text[-2] == " ":
            return True

        return False

    # given AnnotateHtmlNode node class, find children node by html tag. Return list of nodeid for each tag
    def find_node_in_children(self, node, target_node_tag):
        # return dictionary, key: tag value: list of node found
        result = {}
        if len(target_node_tag) == 0:
            return result

        for tag in target_node_tag:
            result[tag] = []

        # use queue for children 
        node_q = queue.Queue()
        node_q.put(node)

        while node_q.qsize() > 0:
            # pop
            current_node = node_q.get()
            
            if (current_node.html_node.name in target_node_tag):
                result[current_node.html_node.name].append(current_node.nodeid)
            
            for child in current_node.children_nodes:
                node_q.put(child)
        
        return result


    def init_vdom_feature(self):
        vdom_feature = self.annotateHtml.VDomFeatures()
        # initialize all feature as 0
        vdom_feature.position_x = 0
        vdom_feature.position_y = 0
        vdom_feature.position_w = 0
        vdom_feature.position_h = 0
        vdom_feature.offset_left = 0
        vdom_feature.offset_top = 0
        vdom_feature.offset_w = 0
        vdom_feature.offset_h = 0
        vdom_feature.client_left = 0
        vdom_feature.client_top = 0
        vdom_feature.client_w = 0
        vdom_feature.client_h = 0
        vdom_feature.font_color_a = 0
        vdom_feature.font_color_r = 0
        vdom_feature.font_color_b = 0
        vdom_feature.font_color_g = 0
        vdom_feature.font_weight = 0
        vdom_feature.font_size = 0
        vdom_feature.font_italic = 0
        vdom_feature.font_decoration = 0
        vdom_feature.list_style = 0
        vdom_feature.display_style = 0
        vdom_feature.cursor_style = 0
        vdom_feature.line_height = 0
        vdom_feature.text_transform = 0
        vdom_feature.opacity = 0
        vdom_feature.border_style_left = 0
        vdom_feature.border_style_top = 0
        vdom_feature.border_style_right = 0
        vdom_feature.border_style_bottom = 0

        return vdom_feature


    def serialize_node_feature_to_array(self, vdom_feature):
        feature_arrary = []
        feature_arrary.append(vdom_feature.position_x)
        feature_arrary.append(vdom_feature.position_y)
        feature_arrary.append(vdom_feature.position_w)
        feature_arrary.append(vdom_feature.position_h)
        feature_arrary.append(vdom_feature.offset_left)
        feature_arrary.append(vdom_feature.offset_top)
        feature_arrary.append(vdom_feature.offset_w)
        feature_arrary.append(vdom_feature.offset_h)
        feature_arrary.append(vdom_feature.client_left)
        feature_arrary.append(vdom_feature.client_top)
        feature_arrary.append(vdom_feature.client_w)
        feature_arrary.append(vdom_feature.client_h)
        feature_arrary.append(vdom_feature.font_color_a)
        feature_arrary.append(vdom_feature.font_color_r)
        feature_arrary.append(vdom_feature.font_color_b)
        feature_arrary.append(vdom_feature.font_color_g)
        feature_arrary.append(vdom_feature.font_weight)
        feature_arrary.append(vdom_feature.font_size)
        feature_arrary.append(vdom_feature.font_italic)
        feature_arrary.append(vdom_feature.font_decoration)
        feature_arrary.append(vdom_feature.list_style)
        feature_arrary.append(vdom_feature.display_style)
        feature_arrary.append(vdom_feature.cursor_style)
        feature_arrary.append(vdom_feature.line_height)
        feature_arrary.append(vdom_feature.text_transform)
        feature_arrary.append(vdom_feature.opacity)
        feature_arrary.append(vdom_feature.border_style_left)
        feature_arrary.append(vdom_feature.border_style_top)
        feature_arrary.append(vdom_feature.border_style_right)
        feature_arrary.append(vdom_feature.border_style_bottom)

        return feature_arrary

    # AnnotateHtml adds additional attributes or nodes to help parse annotations and visual feature
    # This function helps get original html look in string format
    def get_original_html(self):
        original_html = ""

        orig_html_soup = BeautifulSoup(self.htmlStr, 'html.parser')

        all_soup_nodes = orig_html_soup.find_all()

        for node in all_soup_nodes:
            
            # remove special attribute in <html>
            if node.name == "html":
                if "semanticlayout" in node.attrs:
                    del node.attrs["semanticlayout"]
                if "mattrdefaults" in node.attrs:
                    del node.attrs["mattrdefaults"]

            # remove all id attribute
            if "data-dcnode-id" in node.attrs:
                del node.attrs["data-dcnode-id"]

            # remove <span> node added that are not in original html page
                if "instrument_node" in node.attrs:
                    try:
                        node.replaceWith("".join(node.contents))
                    except:
                        del node.attrs["instrument_node"]

        original_html = str(orig_html_soup)
        return original_html

    
    
    
    
    

    
