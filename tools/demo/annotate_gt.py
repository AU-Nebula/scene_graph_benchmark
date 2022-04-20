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
from fusion.utils import *
from re import A, search
from networkx import path_graph
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source

def visualize_step(frame_path, step_json):
    
    '''
    Draw selected STEP action and corresponding bounding-box on frame of choice

    Parameters
    ----------
    
    frame_path : string
        
        File path of the image

    step_json : string

        File path of STEP annotation
    
    action : int
        
        Index corresponding to the selected action
    
    Returns
    -------

    Image file containing STEP annotation
    '''

    # load JSON annotation
    with open(step_json) as file:
        step = json.load(file)
    
    img = cv2.imread(frame_path)
    
    for action in range(len(step['detection_classes'])):
        
        labels = step['detection_classes'][action]
   
        box = step['detection_boxes'][action]

        frame_name = frame_path.split(".")[0]
        output = frame_name + '.step.jpg'

        # read frame of interest
        out = cv2.imread(output)
    
        box = yolo_nebula_to_pixels(box)
    
        # represents the top left corner of rectangle
        start_point = (int(box[0]), int(box[1]))
  
        # represents the bottom right corner of rectangle
        end_point = (int(box[2]), int(box[3]))
   
        # text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (int(box[0]) + 15, int(box[1]) + 30 + action*30)
        fontScale = 1
        f_thickness = 2

        color = (255, 255, 0)
  
        # line thickness
        l_thickness = 4
  
        frame_name = frame_path.split(".")[0]
        output = frame_name + '.step.jpg'

        # write labels 
        text = cv2.putText(img, labels, position, font, fontScale, color, f_thickness, cv2.LINE_AA)
        cv2.imwrite(output, text) 
    
        # draw bounding-box
        rectangle = cv2.rectangle(img, start_point, end_point, color, l_thickness) 
        cv2.imwrite(output, rectangle)
    
    print(output + ' has been exported correctly!')


def annotate_fusion_gt(ctde_json, step_json):

    '''
    Annotate ground truth for STEP-FUSION

    Input
    -----

    ctde_json : string
    step_json : string
    
    Output
    ------

    .json file containing STEP-FUSION ground truth
    '''

    # load STEP annotation
    with open(step_json) as file:
        step = json.load(file)

    # load Casual-TDE annotation
    with open(ctde_json) as file:
        ctde = json.load(file)

    
    action_labels = step['detection_classes']
    action_labels = set(action_labels)
    action_labels = list(action_labels)

    action_boxes = step['detection_boxes']
    action_labels = set(action_labels)
    action_labels = list(action_labels)

    for a in range(len(action_labels)):

        IoU_scores = []
        action_bb = yolo_nebula_to_pixels(action_boxes[a])
        
        actor_id = 0

        for node in ctde['nodes']:
        
            #print(node['id'], node['class'], get_IoU(action_bb, node['bb']))
        
            IoU_bb = get_IoU(action_bb, node['bb'])
        
            if IoU_bb >= 0 and IoU_bb <= 1:
                IoU_scores.append([IoU_bb, node['id'], node['bb'], node['class']])
                IoU_scores = sorted(IoU_scores, reverse=True)
                
                argmax_IoU = IoU_scores[0]
                actor_id = argmax_IoU[1]

        print(argmax_IoU)

        step_edge = {"source": actor_id, "dest": [], "bb": action_bb, "class": [action_labels[a]], "confidence": [], "expert": ["step"]}
        
        # append 'step_edge' to graph "edges" in Causal_TDE dictionary
        ctde['edges'].append(step_edge)
        extended_sg = ctde
    
    # write JSON file
    filename = ctde_json.split('.')[0]
    fusion = '_step-fusion_'
    output = filename + fusion + 'gt.json'
    
    with open(output, 'w') as outfile:
        json.dump(extended_sg, outfile, indent=4, sort_keys=False)
    
    print('\n STEP-FUSION ground truth has been successfully generated!')

    return extended_sg


def sg_to_graphviz(sg_json):

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
        

    node_layout = nx.spring_layout(G)
    nx.draw(G, node_layout, with_labels=True)
    _ = nx.draw_networkx_edge_labels(G, node_layout, edge_labels, font_size = 12)

    dot = to_pydot(G).to_string()
    print(dot)
    skg = Source(dot) # dot is string containing DOT notation of graph
    skg.view()



def isolate_step_gt(S_GT):

    with open(S_GT) as file:
        step_gt = json.load(file)

    nodes = step_gt['nodes']
    edges = step_gt['edges']

    node_labels = {}
    edge_labels = {}

    for v in range(len(nodes)):

        node_id = nodes[v]['id']
        node_label = nodes[v]['class'][0]
        node_labels[node_id] = node_label
    
    step_edges = {}
    step_list = []
    #gt = {'edges': []}
    
    for e in range(len(edges)):

        search_step_edges = re.search(r"step", str(step_gt['edges'][e]))

        if search_step_edges:
            x = search_step_edges
            step_edges[e] = x.string
        
            step_list.append(step_edges[e])

        gt_list = []

        for k in range(len(step_list)):
            
            gt_list.append(ast.literal_eval(step_list[k]))

        gt_edges = gt_list
        tuples = {}

        for j in range(len(gt_edges)):

            gt_score = gt_edges[j]['confidence'][0]

            node_m = gt_edges[j]['source']
            m_label = node_labels[node_m]

            edge_mn = gt_edges[j]['class'][0]
        
            node_n = gt_edges[j]['dest']
            n_label = node_labels[node_n]

            gt_tuple = (gt_score, "{} {} {}".format(m_label, edge_mn, n_label))

            tuples[gt_tuple[1]] = gt_tuple[0]
            t = list(tuples.items())

    return t