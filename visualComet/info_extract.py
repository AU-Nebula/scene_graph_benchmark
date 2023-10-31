import json
import argparse
import zipfile
import os
import re


expression = '[0-9]+'


def process_caption(event):
    new_event = re.sub(expression, 'a person', event, 1)
    # if no number is found after the first number, the second new_event will be the same as the first one
    # it cannot happen that 'another person' is replaced instead just 'person', because that would mean that
    # the first re.sub() did not find any number.
    new_event = re.sub(expression, 'another person', new_event, 1)
    
    new_event = new_event.replace('\u2019', '\'')
    new_event = new_event.replace('\xe0', 'a')
    new_event = new_event.replace('\u201c', '\'')
    new_event = new_event.replace('\u201d', '\'')
    return new_event


def save_ids_file(entries, split, annots_dir):
    json_file = f'{annots_dir}/{split}_prepross.json'
    data = {}
    for entity in entries:
        idx = entity['img_fn']
        event = entity['event']
        caption = process_caption(event)
        if idx not in data:
            data[idx] = {'events': [event], 'captions': [caption]}
        else:
            data[idx]['events'].append(event)
            data[idx]['captions'].append(caption)
    data = [{'idx': idx, 'events': v['events'], 'captions': v['captions']} for idx, v in data.items()]
    with open(json_file, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    pass


def main(args):

    annots_dir = args.dir
    splits = ['train', 'val', 'test']

    # The splits are kept separated
    for split in splits:
        annots_path = f'visualcomet/{split}_annots.json'
        print('****************************************')
        print('{} SPLIT'.format(split.upper()))
        print('****************************************')
        # Accessing and saving all the entries of the document
        with zipfile.ZipFile(f'{annots_dir}{os.sep}visualcomet.zip', 'r') as z:
            with z.open(annots_path) as f:
                print('Retrieving data...')
                entries = json.loads(f.read().decode('utf-8'))

        print('Saving visual ids')
        save_ids_file(entries, split, annots_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Images Indexing')
    
    # Selection directory in which the annotations are stored
    parser.add_argument('--dir', default='visualComet',
                            help='annotation directory path')
    args = parser.parse_args()
    main(args)