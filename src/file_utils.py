import pandas as pd
import os
import json
import csv

import numpy as np

def write_top_file(path:str,reference_names,txt_top_names,img_top_names):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "txt_top_names": txt_top_names.tolist()[count],
                    "img_top_names": img_top_names.tolist()[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)

def write_a_suggestions_file(path:str,reference_name,t2i_suggestion,i2i_suggestion):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = {
        "image_index": reference_name,
        "t2i_suggestion": t2i_suggestion,
        "i2i_suggestion": i2i_suggestion
    }
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return

def write_pseudo_targets_file(path:str,reference_names,pseudo_targets1, confidences1, pseudo_targets2, confidences2):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "pseudo_target1": pseudo_targets1[count],
                    "confidence1": confidences1[count],
                    "pseudo_target2": pseudo_targets2[count],
                    "confidence2": confidences2[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)
    return


def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()  
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj

def write_candidates_file(path: str, reference_name, txt_scores, img_scores):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "image_index": reference_name,
        "T2I": to_serializable(txt_scores),
        "I2I": to_serializable(img_scores)
    }

    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

    return True


def write_a_pseudo_target_file(path: str, reference_name, rank1,rank2,pseudo_target1, confidence1, pseudo_target2, confidence2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = {
        "image_index": reference_name,
        "rank1": rank1,
        "pseudo_target1": pseudo_target1,
        "confidence1": float(confidence1),
        "rank2": rank2,
        "pseudo_target2": pseudo_target2,
        "confidence2": float(confidence2)
    }
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return True

def write_suggestions_file(path:str,reference_names,t2i_suggestions,i2i_suggestions):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "t2i_suggestion": t2i_suggestions[count],
                    "i2i_suggestion": i2i_suggestions[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)
    return


def write_two_suggestions_file(path:str,reference_names,input_suggestions1,input_suggestions2):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "input_suggestions1": input_suggestions1[count],
                    'input_suggestions2': input_suggestions2[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)
    return

def write_edited_images(path,reference_names,target_names,edit_result_imgs):
    edit_img_paths = []
    for i in len(reference_names):
        edit_img_path = os.path.join(path,f"{reference_names[i]}_edited_{target_names[i]}.png")
        edit_result_imgs[i].save(edit_img_path)
        edit_img_paths.append(edit_img_path)
    return edit_img_paths

def write_edited_image(path,reference_name,target_name,edit_result_img):
    edit_img_path = os.path.join(path,f"{reference_name}_edited_{target_name}.png")
    edit_result_img.save(edit_img_path)
    return edit_img_path

def write_modified_captions_file(path:str,reference_names,modified_captions):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "modified_caption": modified_captions[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)
    return

def read_suggestions_file(path:str):
    suggestions = json.load(open(path,'r',encoding='utf-8'))
    if isinstance(suggestions,dict):
        suggestions = [item["suggestion"] for item in suggestions]
    elif isinstance(suggestions, list):
        suggestions_list = [item.get("suggestion") for item in suggestions if "suggestion" in item]   
        if len(suggestions_list) != 0:
            suggestions = suggestions_list
    elif isinstance(suggestions,set):
        suggestions = list(suggestions)
    return suggestions

def read_modified_captions_file(path:str):
    modified_captions = json.load(open(path, 'r',encoding='utf-8'))
    if isinstance(modified_captions,dict):
        modified_captions = list(modified_captions.values())
    elif isinstance(modified_captions, list):
        modified_list = [item.get("modified_caption") for item in modified_captions if "modified_caption" in item]
        if len(modified_list) != 0:
            modified_captions = modified_list
    elif isinstance(modified_captions,set):
        modified_captions = list(modified_captions)
    return modified_captions

def init_folder(dataset_path,task):
    os.makedirs(f'{dataset_path}/task', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}/suggestions', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}/modified_captions', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}/new_captions', exist_ok=True)