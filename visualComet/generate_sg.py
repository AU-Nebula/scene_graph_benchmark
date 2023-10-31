# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import os.path as op
import os
import zipfile

import argparse
import json
import torch
import numpy as np

from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from tools.demo.detect_utils import detect_objects_on_single_image
#from tools.demo.visual_utils import draw_bb, draw_rel


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        label = label.strip()

        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def main():

    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--dir_path", metavar="FILE", help="dir path of visual comet dataset")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--device", default="cuda",
                        help="choose the device you want to work with")
    parser.add_argument("--visualize_attr", action="store_true",
                        help="visualize the object attributes")
    parser.add_argument("--visualize_relation", action="store_true",
                        help="visualize the relationships")
    parser.add_argument("--min_obj_score", metavar="OBJECTS THRESHOLD", type=restricted_float, default=0,
                    	help="threshold to filter objects generation")
    parser.add_argument("--min_rel_score", metavar="RELATIONSHIPS THRESHOLD", type=restricted_float, default=0,
                        help="threshold to filter relationships generation")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device=="cuda":
        raise RuntimeError("No GPU available. Please check the device selected or set up again the software by following the steps in the README file in section 1b")

    cfg.MODEL.DEVICE = args.device
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    # visual_labelmap is used to select classes for visualization
    try:
        visual_labelmap = load_labelmap_file(args.labelmap_file)
    except:
        visual_labelmap = None

    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        dataset_attr_labelmap = {
            int(val): key for key, val in
            dataset_allmap['attribute_to_idx'].items()}

    if cfg.MODEL.RELATION_ON and args.visualize_relation:
        dataset_relation_labelmap = {
            int(val): key for key, val in
            dataset_allmap['predicate_to_idx'].items()}

    transforms = build_transforms(cfg, is_train=False)

    paths = set()
    for prepross in ['train', 'test', 'val']:
        f = f'{args.dir_path}{os.sep}{prepross}_prepross.json'
        with open(f, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        paths.update(map(lambda x: x['idx'], data))
    import time
    start = time.time()
    with zipfile.ZipFile(f'{args.dir_path}{os.sep}vcr1images.zip', 'r', allowZip64=True) as z:
        for i, path in enumerate(paths):
            if (i % 100) == 0:
                print(f'{i} of {len(paths)}... {(i * 100 / len(paths)):.2f}% - {time.time()-start} s.') 
            with z.open(f'vcr1images/{path}', 'r') as f:
                try:
                    buffer = np.asarray(bytearray(f.read()), dtype="uint8")
                except zipfile.BadZipFile:
                    print(f'Fail at open: {path} -- skiping')
                    continue 
            process_image(buffer, args, model, transforms, dataset_labelmap, visual_labelmap, dataset_attr_labelmap, dataset_relation_labelmap, path, args.dir_path)
    pass


def process_image(buffer, args, model, transforms, dataset_labelmap, visual_labelmap, dataset_attr_labelmap, dataset_relation_labelmap, image_path, dirpath):
    path = f'{dirpath}{os.sep}sg{os.sep}{op.dirname(image_path)}'
    save_file = path + os.sep + op.splitext(op.basename(image_path))[0] + ".causal_tde.json"

    if op.exists(save_file):
        return
    
    cv2_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    dets = detect_objects_on_single_image(model, transforms, cv2_img)

    if isinstance(model, SceneParser):
        rel_dets = dets['relations']
        dets = dets['objects']

    for obj in dets:
        obj["class"] = dataset_labelmap[obj["class"]]
        obj["class"] = obj["class"].strip()
    
    if visual_labelmap is not None:
        dets = [d for d in dets if d['class'] in visual_labelmap]
    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        for obj in dets:
            obj["attr"], obj["attr_conf"] = postprocess_attr(dataset_attr_labelmap, obj["attr"], obj["attr_conf"])

    if cfg.MODEL.RELATION_ON and args.visualize_relation:
        for rel in rel_dets:
            rel['class'] = dataset_relation_labelmap[rel['class']]
            subj_rect = dets[rel['subj_id']]['rect']
            rel['subj_center'] = [(subj_rect[0]+subj_rect[2])/2, (subj_rect[1]+subj_rect[3])/2]
            obj_rect = dets[rel['obj_id']]['rect']
            rel['obj_center'] = [(obj_rect[0]+obj_rect[2])/2, (obj_rect[1]+obj_rect[3])/2]


    rects = [d["rect"] for d in dets]
    scores = [d["conf"] for d in dets]

    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        attr_labels = [','.join(d["attr"]) for d in dets]
        attr_scores = [d["attr_conf"] for d in dets]
        labels = [attr_label+' '+d["class"]
                  for d, attr_label in zip(dets, attr_labels)]
    else:
        labels = [d["class"] for d in dets]

    #draw_bb(cv2_img, rects, labels, scores)
    
    graph = {"frame": image_path,
             "nodes":[],
             "edges":[],
             "lighthouse":[]
	    }

    accepted_nodes = set()

    for id,rect in enumerate(rects):
        if scores[id] <= args.min_obj_score:
            continue
        accepted_nodes.add(id)
        node = {"id": id, "bb": rect, "kg_mapping":[], "class": [labels[id].strip()], "confidence":[scores[id]], "expert": ["causal_tde"]}
        graph["nodes"].append(node)

    # merge(dets[rel['subj_id']]['rect'], dets[rel['obj_id']]['rect'])

    for rel in rel_dets:
        if rel['conf'] <= args.min_rel_score:
            continue
        if rel["subj_id"] in accepted_nodes and rel["obj_id"] in accepted_nodes:
             edge = {"source": rel["subj_id"], "dest": rel["obj_id"], "bb": [], "class": [rel["class"]], "confidence": [rel["conf"]], "expert": ["causal_tde"]}
             graph["edges"].append(edge)

    if not op.exists(path):
        os.makedirs(path)
    # save results in json format
    with open(save_file, 'wt', encoding='utf-8') as f:
        json.dump(graph, f, indent=4)


if __name__ == "__main__":
    main()
