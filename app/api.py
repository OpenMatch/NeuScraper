# Copyright (c) OpenMatch. All rights reserved.
# See LICENSE file in the project root for license information.

class CommonCrawlApi:
    class AnnotateHtmlNode:
        def __init__(self, id):
            self.nodeid = id
            self.html_node = None
            self.is_textnode = False
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


    def __init__(self, init_nodes=True, html_soup=None):

        if len(html_soup.prettify()) > 0:
            self.soup = html_soup

        self.all_nodes = {}
        self.nodeid_to_feature = {}
        self.textnode_id_in_order = []
        self.html_title_node = None

        #initialize node class and relationship
        if init_nodes and len(html_soup.prettify().strip()) > 0:
            self.init_all_nodes()



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


