#!/bin/python
# Author: Francesco Maria Turno

# This file contains functions that might be helpful for annotating STEP-FUSION ground truth 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json, re, ast
import pydot
import graphviz
import argparse

from re import A, search
from networkx import path_graph
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source

def sg_to_graphviz(sg_json, out_path=None):

    '''
    Export Scene Graph to GraphViz format     

    Input
    -----

    sg_json : string
        
        File path of JSON file containing Scene Graph

    Output
    ------

    gdot : composite data type 
    
        Graph object in .dot format 
    '''
    
    # load JSON annotation containing extended SG
    with open(sg_json) as file:
        sg = json.load(file)

    # initialize Graph constructor
    G = nx.MultiDiGraph()
    
    # add Graph edges
    links = sg['edges'] 
    vertices = sg['nodes']

    # print(vertices)

    edge_labels = {}
    node_labels = {}
    node_confidences = {}

    for i, v in enumerate(vertices):

        node_id = vertices[i]['id']

        node_label = vertices[i]['class'][0]
        
        node_conf = vertices[i]['confidence'][0]
        node_conf = round(node_conf, 2)
        
        node_labels[node_id] = node_label
        node_confidences[node_id] = node_conf

    connected_nodes_id = []

    for i, e  in enumerate(links):
        
        node_i = links[i]['source']
        node_j = links[i]['dest']
        
        edge_weight = links[i]['confidence'][0]
        edge_weight = round(edge_weight, 2)
        
        if "step" in links[i]['expert']:
            rel = "** {} **".format(links[i]['class'][0]) + '\n' + str(edge_weight)
        else:
            rel = links[i]['class'][0] + '\n' + str(edge_weight)

        s_label = node_labels[node_i]
        d_label = node_labels[node_j]

        s_conf = str(node_confidences[node_i])
        d_conf = str(node_confidences[node_j])

        G.add_edge("({}) {} = {}".format(node_i, s_label, s_conf), "({}) {} = {}".format(node_j, d_label, d_conf), label = rel)

        s_key = "({}) {} = {}".format(node_i, s_label, s_conf)
        d_key = "({}) {} = {}".format(node_j, d_label, d_conf)

        _key = (s_key, d_key)
        
        if _key in edge_labels:
            edge_labels[_key] += " / " + rel + str(edge_weight)
        else:
            edge_labels[_key] = rel + str(edge_weight)
        
        connected_nodes_id.append(node_i)
        connected_nodes_id.append(node_j)


    isolated_nodes_id = sorted(list(set(node_labels.keys())-set(connected_nodes_id)))

    for node_id in isolated_nodes_id:
        node_i = node_id
        s_label = node_labels[node_i]
        s_conf = node_confidences[node_i]

        isolated_node = "({}) {} = {}".format(node_i, s_label, s_conf)

        G.add_node(isolated_node)

    node_layout = nx.spring_layout(G)
    nx.draw(G, node_layout, with_labels=True)
    _ = nx.draw_networkx_edge_labels(G, node_layout, edge_labels, font_size = 12)

    dot = to_pydot(G).to_string()
    skg = Source(dot) # dot is string containing DOT notation of graph
    if not out_path:
        skg.render(sg_json.replace(".causal_tde.json","_graph"), view=False, cleanup=True, format='pdf')
    else:
        skg.render(out_path, view=False, cleanup=True, format='pdf')

def main():

    parser = argparse.ArgumentParser(description="Graphviz generation Demo")
    parser.add_argument("--img_file",
                        help="path to json annotation file")
    parser.add_argument("--save_file",
                        help="filename to save the image graph")
    
    args = parser.parse_args()
    
    sg_to_graphviz(args.img_file, args.save_file)

if __name__ == "__main__":
    main()