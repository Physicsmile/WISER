import json
import os
from typing import Optional, Tuple, List, Dict, Union

import argparse
import clip
import numpy as np
import openai_api
import pickle
import torch
import tqdm
import datasets
import data_utils
import prompts
import datetime
import pandas as pd
import re
import data_utils

from check_prompt import CheckModel, ModelHandler
from get_pseudo_targets import VQAModelHandler
import csv
import file_utils
from PIL import Image
from bagel_inference import BagelImageEditor

if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


@torch.no_grad()
def extract_image_features(device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 8, preload: str=None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        index_ranks = [] if 'index_ranks' not in extracted_data else extracted_data['index_ranks']        
        aux_data = {} if 'aux_data' not in extracted_data else extracted_data['aux_data']
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)

        index_features, index_names, index_ranks, aux_data = [], [], [], []
        # if 'genecis' in args.dataset:
        #     aux_data = {'ref_features': [], 'instruct_features': []}
            
        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features    
        index_rank = None
        for batch in tqdm.tqdm(loader):

            images = batch.get('image')
            names = batch.get('image_name')
            if images is None: images = batch.get('reference_image')
            if names is None: names = batch.get('reference_name')

            images = images.to(device)
            with torch.no_grad(),torch.cuda.amp.autocast():
                batch_features = clip_model.encode_image(images)
                index_features.append(batch_features.cpu())
                index_names.extend(names)
                if index_rank is not None:
                    index_ranks.extend(index_rank)
                if len(aux_data):
                    aux_data['ref_features'].append(clip_model.encode_image(ref_images.to(device)).cpu())
                    if hasattr(clip_model, 'tokenizer'):
                        aux_data['instruct_features'].append(clip_model.encode_text(clip_model.tokenizer(instructions, context_length=77).to(device)).cpu())
                    else:
                        aux_data['instruct_features'].append(clip_model.encode_text(clip.tokenize(instructions, context_length=77).to(device)).cpu())
        
        index_features = torch.vstack(index_features)


        if preload is not None:
            os.makedirs(os.path.dirname(preload), exist_ok=True)
            pickle.dump({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks, 'aux_data': aux_data}, open(preload, 'wb'))
            print(f"Save image feathers in {preload}")
    return index_features, index_names, index_ranks, aux_data


@torch.no_grad()
def generate_editimg_caption_iteration(
    device: torch.device, args: argparse.Namespace, bagel_editor:BagelImageEditor,dataset_name:str,llm_prompt_args: str,retrieval:str,clip_model: clip.model.CLIP,
    query_dataset: torch.utils.data.Dataset, target_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]],processor,LLM_model_name,max_check_num, 
    Check_LLM_model_name,VQA_LLM_model_name,dataset_path,edit_img_dir,compute_results_function,index_features,index_names,openai_key,task,split,preprocess,**kwargs
) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions
    """    
    ### Generate Edited Captions.
    torch.cuda.empty_cache()    
    batch_size = 4
    reload_caption_dict = {}
    reload_img_paths_dict = {}

    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        assert "Must generate initial captions before!"
    else:
        print(f'Loading precomputed image captions from {preload_dict["captions"]}!')
        all_captions, relative_captions = [], []
        ref_img_paths = []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []
        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=4, 
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Loading image captions...')

        with open(preload_dict['captions'], 'r', encoding='utf-8') as blip_captions:
            reader = csv.reader(blip_captions)
            next(reader)
            reload_caption_dict = {caption[0].lstrip('0'): caption[1] for caption in reader}

        with open(preload_dict['img_paths'], 'r', encoding='utf-8') as img_paths:
            reader = csv.reader(img_paths)
            next(reader)
            reload_img_paths_dict = {img_path[0].lstrip('0'): img_path[1] for img_path in reader}         
        
        for batch in query_iterator:
            if 'genecis' in dataset_name:
                relative_captions.extend(batch[1])
            else:
                reference_names.extend(batch['reference_name'])
                if 'fashioniq' not in dataset_name:
                    relative_captions.extend(batch['relative_caption'])
                else:
                    rel_caps = batch['relative_captions']
                    rel_caps = np.array(rel_caps).T.flatten().tolist()
                    relative_captions.extend([
                        f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)
                        ])
                                
                if 'target_name' in batch:
                    target_names.extend(batch['target_name'])
            
                gt_key = 'gt_img_ids'
                if 'group_members' in batch:
                    gt_key = 'group_members'
                if gt_key in batch:
                    gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

                query_key = 'query_id'
                if 'pair_id' in batch:
                    query_key = 'pair_id'
                if query_key in batch:
                    query_ids.extend(batch[query_key])
            # match captions and target images
            for target_name in batch['reference_name']:
                # print("target_name:",target_name)
                all_captions.append(reload_caption_dict[target_name])
                ref_img_paths.append(reload_img_paths_dict[target_name])
            
    ### Modify Captions using LLM.
    suggestions = [''] * len(all_captions)
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        modified_captions = LLM_modify_editimg_caption(bagel_editor,LLM_model_name,preload_dict,llm_prompt_args,all_captions,relative_captions,openai_key,device)
        preload_dict['mods']= f'{dataset_path}/task/{task}/modified_captions/{dataset_name}_modified_captions.json'
        file_utils.write_modified_captions_file(preload_dict['mods'],reference_names=reference_names,modified_captions=modified_captions)
    else:
        print(f'Loading precomputed caption modifiers from {preload_dict["mods"]}!')
        modified_captions = file_utils.read_modified_captions_file(path=preload_dict["mods"])
    
    ### Edited Imgs using Bagel.
    if preload_dict['edit_images'] is None or not os.path.exists(preload_dict['edit_images']):
        all_edit_img_paths = []
        for batch in tqdm.tqdm(query_iterator, desc='Editing images'): 
            if 'genecis' in dataset_name:
                relative_captions.extend(batch[1])
            else:
                reference_names.extend(batch['reference_name'])
                if 'fashioniq' not in dataset_name:
                    relative_captions.extend(batch['relative_caption'])
                else:
                    rel_caps = batch['relative_captions']
                    rel_caps = np.array(rel_caps).T.flatten().tolist()
                    relative_captions.extend([
                        f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)
                        ])
                                
                if 'target_name' in batch:
                    target_names.extend(batch['target_name'])
            
                gt_key = 'gt_img_ids'
                if 'group_members' in batch:
                    gt_key = 'group_members'
                if gt_key in batch:
                    gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

                query_key = 'query_id'
                if 'pair_id' in batch:
                    query_key = 'pair_id'
                if query_key in batch:
                    query_ids.extend(batch[query_key])


            edit_img_paths = []
            for i in range(len(batch['reference_image_path'])):
                img_path = batch['reference_image_path'][i]
                text = batch['relative_caption'][i]

                img_name = os.path.basename(img_path)
                base_name, ext = os.path.splitext(img_name)
                mid_path = batch[query_key][i] if query_key in batch else batch['target_name'][i]
                edit_img_name = f"{base_name}_edited_{mid_path}{ext}"
                edit_img_path = os.path.join(edit_img_dir, edit_img_name)

                if os.path.exists(edit_img_path):
                    try:
                        edit_img_paths.append(edit_img_path)

                    except Exception as e:
                        print(f"加载现有文件失败: {e}")
                        continue
                else:
                    edit_result = bagel_editor.edit_image_no_think(img_path, text)
                    edit_result_img = edit_result['image']
                    edit_result_img.save(edit_img_path)
                    edit_img_paths.append(edit_img_path)

            all_edit_img_paths.extend(edit_img_paths)

        if preload_dict['edit_images'] is not None:
            res_dict = {
                'all_edit_img_paths': all_edit_img_paths, 
                'gt_img_ids': gt_img_ids, 
                'relative_captions': relative_captions,
                'target_names': target_names,
                'reference_names': reference_names,
                'query_ids': query_ids
            }
            pickle.dump(res_dict, open(preload_dict['edit_images'], 'wb'))
            imgs_res_dict = pickle.load(open(preload_dict['edit_images'], 'rb'))
    else:
        print(f'Loading precomputed images from {preload_dict["edit_images"]}!')
        imgs_res_dict = pickle.load(open(preload_dict['edit_images'], 'rb'))
    
    all_edit_img_paths = imgs_res_dict['all_edit_img_paths']

    ### Retrieval Top-N candidates
    edit_image_dataset = datasets.EditedImageDataset(all_edit_img_paths, preprocess)
    edit_features, edit_names, edit_ranks, edit_aux_data = extract_image_features(
            device, args, edit_image_dataset, clip_model, 
            preload=None 
    )
    predicted_img_features = torch.nn.functional.normalize(edit_features.float(), dim=-1)
    predicted_txt_features = text_encoding(device, clip_model, modified_captions, batch_size, retrieval)

    if 'fashion' in dataset_name:
        txt_output_metrics,txt_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_txt_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path,task=task,ways='t2i') 
        img_output_metrics,img_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_img_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path,task=task,ways='i2i') 
    elif 'cirr' in dataset_name:
        txt_output_metrics,txt_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_txt_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split,ways='t2i')  
        img_output_metrics,img_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_img_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split,ways='i2i')   
    else:
        txt_output_metrics,txt_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_txt_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split,ways='t2i')  
        img_output_metrics,img_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_img_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split,ways='i2i')    
    

    txt_top_names = txt_sorted_index_names[:,:50]
    img_top_names = img_sorted_index_names[:,:50]
    file_utils.write_top_file(f"{dataset_path}/task/{task}/top_rank_loop_0_{task}_{get_time()}.json",reference_names = reference_names,txt_top_names = txt_top_names, img_top_names=img_top_names)

    txt_check_index = [True] * len(modified_captions)
    img_check_index = [True] * len(modified_captions)

    if preload_dict['suggestions'] != None and os.path.exists(preload_dict['suggestions']):
        path1_scores = []
        path2_scores = []
        json_suggestions = json.load(open(preload_dict['suggestions'], 'r'))
        t2i_suggestions = [item['t2i_suggestion'] for item in json_suggestions]
        i2i_suggestions = [item['i2i_suggestion'] for item in json_suggestions]
    
    ranks1 = []
    ranks2 = []

    for i in range(max_check_num):
        print(f"{i} times:Start check modified captions & edited imgs and generate suggestions.")
        # load top image caption
        txt_top_captions = []
        txt_top_img_paths = []
        for names in txt_top_names:
            txt_top_captions_list = []
            txt_top_img_pths_list = []
            for name in names:
                txt_top_captions_list.append(reload_caption_dict[name])
                txt_top_img_pths_list.append(reload_img_paths_dict[name])
            txt_top_captions.append(txt_top_captions_list) 
            txt_top_img_paths.append(txt_top_img_pths_list) 

        img_top_captions = []
        img_top_img_paths = []
        for names in img_top_names:
            img_top_captions_list = []
            img_top_img_pths_list = []
            for name in names:
                img_top_captions_list.append(reload_caption_dict[name])
                img_top_img_pths_list.append(reload_img_paths_dict[name])
            img_top_captions.append(img_top_captions_list)     
            img_top_img_paths.append(img_top_img_pths_list)


        print(f"Start check modified captions...")

        candidates1,candidates2, ranks1,ranks2,pseudo_targets1, confidences1, pseudo_targets2, confidences2, txt_check_index,img_check_index = get_pseudo_targets(Check_LLM_model_name,openai_key,dataset_path,task,i+1,reference_names,VQA_LLM_model_name,txt_top_captions,img_top_captions,txt_top_img_paths,img_top_img_paths,all_captions,ref_img_paths,relative_captions,device=device,txt_check_index=txt_check_index,img_check_index=img_check_index)

        pseudo_target_names1 = [os.path.splitext(os.path.basename(path))[0].lstrip('0') for path in pseudo_targets1]
        pseudo_target_names2 = [os.path.splitext(os.path.basename(path))[0].lstrip('0') for path in pseudo_targets2]
        pseudo_target_captions1 = [reload_caption_dict[name] for name in pseudo_target_names1]
        pseudo_target_captions2 = [reload_caption_dict[name] for name in pseudo_target_names2]

        if preload_dict['suggestions'] is None or not os.path.exists(preload_dict['suggestions']):
            t2i_suggestions,i2i_suggestions = check_prompt(dataset_path,task,i+1,reference_names,Check_LLM_model_name,pseudo_target_captions1,pseudo_target_captions2,all_captions,ref_img_paths,relative_captions,openai_key,device=device,txt_check_index=txt_check_index,img_check_index=img_check_index)
            file_utils.write_suggestions_file(f'{dataset_path}/task/{task}/suggestions/blip2_t5_{Check_LLM_model_name}_suggestions_loop_{i+1}_{task}_{get_time()}.json',reference_names=reference_names, t2i_suggestions=t2i_suggestions,i2i_suggestions=i2i_suggestions)
        else:
            json_suggestions = json.load(open(preload_dict['suggestions'], 'r'))
            t2i_suggestions = [item['t2i_suggestion'] for item in json_suggestions]
            i2i_suggestions = [item['i2i_suggestion'] for item in json_suggestions]
        
        print(f"{i} times:Start remodified captions with suggestions.")

        edited_image_dir = f'{dataset_path}/task/{task}/new_images/{Check_LLM_model_name}_bagel_suggestions_edited_loop_{i+1}_{task}'
        os.makedirs(edited_image_dir, exist_ok=True)

        modified_captions, all_edit_img_paths, txt_check_index, img_check_index, input_suggestions1, input_suggestions2, path1_scores, path2_scores = LLM_remodify_editimg_caption(bagel_editor=bagel_editor,LLM_model_name=LLM_model_name,llm_prompt_args=llm_prompt_args,last_captions=modified_captions,last_img_pths=all_edit_img_paths,ref_img_paths=ref_img_paths, all_captions=all_captions,
                            relative_captions=relative_captions,t2i_suggestions=t2i_suggestions,i2i_suggestions=i2i_suggestions,openai_key=openai_key,device=device,txt_check_index=txt_check_index,img_check_index=img_check_index, reference_names=reference_names, target_names=target_names,query_ids=query_ids, edited_image_dir=edited_image_dir,loop=i+1, task=task, dataset_path=dataset_path)
        file_utils.write_two_suggestions_file(path=f"{dataset_path}/task/{task}/suggestions/blip2_t5_{Check_LLM_model_name}_input_suggestions_loop_{i+1}_{task}_{get_time()}.json",reference_names=reference_names,input_suggestions1 = input_suggestions1,input_suggestions2 = input_suggestions2)
        file_utils.write_modified_captions_file(f'{dataset_path}/task/{task}/new_captions/blip2_t5_{Check_LLM_model_name}_suggestions_modified_captions_loop_{i+1}_{task}_{get_time()}.json',reference_names=reference_names,modified_captions=modified_captions)

        edit_image_dataset = datasets.EditedImageDataset(all_edit_img_paths, preprocess)
        edit_features, edit_names, edit_ranks, edit_aux_data = extract_image_features(
                device, args, edit_image_dataset, clip_model, 
                preload=preload_dict.get('edit_img_features')  
        )
        predicted_img_features = torch.nn.functional.normalize(edit_features.float(), dim=-1)
        predicted_txt_features = text_encoding(device, clip_model, modified_captions, batch_size=batch_size, mode=retrieval)


        if 'fashion' in dataset_name:
            txt_output_metrics,txt_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_txt_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path,task=task,loop=i+1,ways='t2i')   
            img_output_metrics,img_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_img_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path,task=task,loop=i+1,ways='i2i')    
        elif 'cirr' in dataset_name:
            txt_output_metrics,txt_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_txt_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split,task=task,loop=i+1,ways='t2i')  
            img_output_metrics,img_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_img_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split,task=task,loop=i+1,ways='i2i')   
        elif 'circo' in dataset_name:
            txt_output_metrics,txt_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_txt_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split,loop=i+1,task=task,ways='t2i')  
            img_output_metrics,img_sorted_index_names=compute_results_function(device=device,predicted_features=predicted_img_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split,loop=i+1,task=task,ways='i2i')    
        

        txt_top_names = txt_sorted_index_names[:,:50]
        img_top_names = img_sorted_index_names[:,:50]
        file_utils.write_top_file(f"{dataset_path}/task/{task}/top_rank_loop_{i+1}_{task}_{get_time()}.json",reference_names = reference_names,txt_top_names = txt_top_names, img_top_names=img_top_names)
        
        txt_top_captions = []
        txt_top_img_paths = []
        for names in txt_top_names:
            txt_top_captions_list = []
            txt_top_img_pths_list = []
            for name in names:
                txt_top_captions_list.append(reload_caption_dict[name])
                txt_top_img_pths_list.append(reload_img_paths_dict[name])
            txt_top_captions.append(txt_top_captions_list) 
            txt_top_img_paths.append(txt_top_img_pths_list) 

        img_top_captions = []
        img_top_img_paths = []
        for names in img_top_names:
            img_top_captions_list = []
            img_top_img_pths_list = []
            for name in names:
                img_top_captions_list.append(reload_caption_dict[name])
                img_top_img_pths_list.append(reload_img_paths_dict[name])
            img_top_captions.append(img_top_captions_list)     
            img_top_img_paths.append(img_top_img_pths_list)
        txt_scores, img_scores, ranks1,ranks2, pseudo_targets1, confidences1, pseudo_targets2, confidences2, txt_check_index,img_check_index = get_pseudo_targets(Check_LLM_model_name,openai_key,dataset_path,task,i+2,reference_names,VQA_LLM_model_name,txt_top_captions,img_top_captions,txt_top_img_paths,img_top_img_paths,all_captions,ref_img_paths,relative_captions,device=device,txt_check_index=txt_check_index,img_check_index=img_check_index)

    return {
        'predicted_img_features': predicted_img_features, 
        'predicted_txt_features': predicted_txt_features, 
        'ranks1': ranks1,
        'ranks2': ranks2,
        'candidates1': candidates1,
        'candidates2': candidates2,
        'confidences1':confidences1,
        'confidences2':confidences2,
        'target_names': target_names, 
        'targets': gt_img_ids,
        'reference_names': reference_names,
        'query_ids': query_ids,
        'start_captions': all_captions,
        'modified_captions': modified_captions,
        'instructions': relative_captions,
        'loop': max_check_num+1,
    }

def get_time()->str:
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d-%H_%M_%S")

def extract_suggestions(original_suggestion):
    suggestion_split = original_suggestion.split('\n')
    total_suggestion = ""
    suggestion_flag = False
    for line in suggestion_split:
        if suggestion_flag:
            total_suggestion = total_suggestion+line
        elif 'suggestion:' in line or 'Suggestion:' in line:
            suggestion_flag = True
            total_suggestion = total_suggestion + line.split('uggestion:')[1].strip()
    return total_suggestion   

def LLM_remodify_editimg_caption(bagel_editor,LLM_model_name,llm_prompt_args,last_captions,last_img_pths,ref_img_paths,all_captions,relative_captions,t2i_suggestions,i2i_suggestions,openai_key,device:torch.device,txt_check_index,img_check_index, reference_names, target_names,query_ids,edited_image_dir,loop,task,dataset_path):
    modified_captions = []
    edit_img_paths = []
    base_prompt = eval(llm_prompt_args)
    input_suggestions1 = []
    input_suggestions2 = []

    path1_scores = []
    path2_scores = []

    num_need_text_modify = 0
    num_need_image_modify = 0

    if LLM_model_name == "bagel":
        for i in tqdm.trange(len(all_captions), position=1, desc=f'Remodifying captions with LLM...'):
            t2i_total_suggestions = extract_suggestions(t2i_suggestions[i])
            i2i_total_suggestions = extract_suggestions(i2i_suggestions[i])
            if not t2i_total_suggestions:
                t2i_total_suggestions = t2i_suggestions[i]
            if not i2i_total_suggestions:
                i2i_total_suggestions = i2i_suggestions[i]

            need_text_modify = (txt_check_index[i] and 
                            "Good retrieval, no more loops needed" not in t2i_total_suggestions and 
                            t2i_total_suggestions != "")
            
            need_image_modify = (img_check_index[i] and 
                                "Good retrieval, no more loops needed" not in i2i_total_suggestions and 
                                i2i_total_suggestions != "")
            
            if need_text_modify:
                num_need_text_modify+=1
                cleaned_relative = relative_captions[i].strip('.?, ')
                cleaned_suggestions = t2i_total_suggestions.strip('.?,"\' ')  

                Instruction = f"{cleaned_relative} and {cleaned_suggestions}."

                # print("Instruction:",Instruction)
                input_suggestions1.append(t2i_total_suggestions)
                final_prompt = f'''
                {base_prompt}
                Image Content: {all_captions[i]}.
                Instruction: {Instruction}.
                '''
                resp = bagel_editor.generate_caption(final_prompt)
                
                resp = resp.split('\n')
                description = ""
                for line in resp:                    
                    if line.strip().startswith('Edited Description:'):
                        description = line.split(':')[1].strip()
                        break
                # print("description:",description)
                modified_captions.append(description if description else last_captions[i])
                txt_check_index[i] = True
            else:
                input_suggestions1.append("Good retrieval, no more loops needed")
                modified_captions.append(last_captions[i])
                txt_check_index[i] = False
            
            if need_image_modify:
                num_need_image_modify+=1
                input_suggestions2.append(i2i_total_suggestions)
                
                if not target_names:
                    target_identifier = query_ids[i]
                else:
                    target_identifier = target_names[i]
                    
                tmp_pth = os.path.join(edited_image_dir, f"{reference_names[i]}_edited_{target_identifier}.png")
                
                if os.path.exists(tmp_pth):
                    edit_img_path = tmp_pth
                else:
                    cleaned_relative = relative_captions[i].strip('.?, ')
                    cleaned_suggestions = i2i_total_suggestions.strip('.?,"\' ')  

                    i2i_prompt = f"{cleaned_relative} and {cleaned_suggestions}."    
                    # print("i2i_prompt:",i2i_prompt)
                    edit_result = bagel_editor.edit_image_no_think(ref_img_paths[i], i2i_prompt)
                    edit_result_img = edit_result['image']
                    edit_img_path = file_utils.write_edited_image(
                        edited_image_dir, 
                        reference_name=reference_names[i], 
                        target_name=target_identifier, 
                        edit_result_img=edit_result_img
                    )
                
                edit_img_paths.append(edit_img_path)
                img_check_index[i] = True
            else:
                input_suggestions2.append("Good retrieval, no more loops needed")
                
                if not target_names:
                    target_identifier = query_ids[i]
                else:
                    target_identifier = target_names[i]
                    
                expected_img_path = os.path.join(edited_image_dir, f"{reference_names[i]}_edited_{target_identifier}.png")
                
                if os.path.exists(expected_img_path):
                    edit_img_paths.append(expected_img_path)
                else:
                    source_path = last_img_pths[i]
                    if os.path.exists(source_path):
                        import shutil
                        shutil.copy2(source_path, expected_img_path)
                        print(f"Copied image from {source_path} to {expected_img_path}")
                        edit_img_paths.append(expected_img_path)
                    else:
                        edit_img_paths.append(source_path)
                
                img_check_index[i] = False

        return modified_captions, edit_img_paths, txt_check_index, img_check_index, input_suggestions1, input_suggestions2, path1_scores, path2_scores


def LLM_modify_editimg_caption(bagel_editor,LLM_model_name, preload_dict,llm_prompt_args,all_captions,relative_captions,openai_key,device:torch.device): 
    modified_captions = []
    base_prompt = eval(llm_prompt_args)
    if LLM_model_name == "bagel":
        for i in tqdm.trange(len(all_captions), position=1, desc=f'Modifying captions with LLM...'):
            instruction = relative_captions[i]
            img_caption = all_captions[i]
            final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
            final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction

            resp = bagel_editor.generate_caption(final_prompt)
            resp = resp.split('\n')
            description = ""
            aug = False
            for line in resp:                    
                if line.strip().startswith('Edited Description:'):
                    description = line.split(':')[1].strip()
                    if description == "":
                        modified_captions.append(relative_captions[i])
                    else:
                        modified_captions.append(description)
                    aug = True
                    break
            if not aug:
                modified_captions.append(relative_captions[i])   
        return modified_captions

def check_prompt(dataset_path,task,loop,reference_names,model_name,txt_top_captions,img_top_captions,all_captions,ref_img_paths,relative_captions,openai_key,txt_check_index,img_check_index,device:torch.device):
    model_handler = ModelHandler(model_type=model_name, device=device, openai_key=openai_key)

    t2i_suggestions = []
    i2i_suggestions = []
    count=0
    for i in tqdm.trange(648,len(relative_captions), position=1, desc='Generate suggestions with LLM...'):
        t2i_suggestion = ''
        i2i_suggestion = ''
        if txt_check_index[i] == True:
            t2i_suggestion = model_handler.chat_function("t2i", all_captions[i], relative_captions[i], txt_top_captions[i], img_top_captions[i], device, max_length=10000)
        if img_check_index[i] == True:
            i2i_suggestion = model_handler.chat_function("i2i", all_captions[i], relative_captions[i], txt_top_captions[i], img_top_captions[i], device, max_length=10000)

        if t2i_suggestion is None:
            t2i_suggestion = ''
        if i2i_suggestion is None:
            i2i_suggestion = ''

        t2i_suggestions.append(t2i_suggestion)
        i2i_suggestions.append(i2i_suggestion)
    return t2i_suggestions,i2i_suggestions


def get_pseudo_targets(Check_LLM_model_name,openai_key,dataset_path,task,loop,reference_names,model_name, txt_top_captions, img_top_captions, txt_top_img_paths,img_top_img_paths,all_captions, ref_img_paths,relative_captions, txt_check_index, img_check_index, device:torch.device):
    model_handler = VQAModelHandler(model_type=model_name, device=device,openai_key=openai_key)

    pseudo_targets1 = []
    pseudo_targets2 = []
    confidences1 = []
    confidences2 = []
    threshold = 0.7 
    ranks1 = []
    ranks2 = []
    candidates1 = []
    candidates2 = []
    start_index = 0
    for i in tqdm.trange(start_index, len(relative_captions), position=1, desc='Scoring candidates with VQA...'):
        txt_scores = []
        for j, candidate_path in enumerate(txt_top_img_paths[i]):
            confidence = model_handler.chat_function(
                ref_img_paths[i], 
                relative_captions[i],  
                candidate_path, 
                device
            )
            txt_scores.append((j+1, candidate_path, confidence))
        
        img_scores = []
        for j, candidate_path in enumerate(img_top_img_paths[i]):
            confidence = model_handler.chat_function(
                ref_img_paths[i],
                relative_captions[i], 
                candidate_path,
                device
            )
            img_scores.append((j+1, candidate_path, confidence))
        candidates1.append(txt_scores)
        candidates2.append(img_scores)

        if len(txt_scores) > 0:
            best_txt = max(txt_scores, key=lambda x: x[2])  
            rank1, pseudo_target1, confidence1 = best_txt[0], best_txt[1], best_txt[2]
        else:
            rank1, pseudo_target1, confidence1 = None, None, 0.0

        if len(img_scores) > 0:
            best_img = max(img_scores, key=lambda x: x[2])
            rank2, pseudo_target2, confidence2 = best_img[0], best_img[1], best_img[2]
        else:
            rank2, pseudo_target2, confidence2 = None, None, 0.0
        file_utils.write_candidates_file(f'{dataset_path}/task/{task}/pseudo_targets/{Check_LLM_model_name}_blip2_t5_{model_name}_candidates_loop_{loop}_{task}.json',reference_name=reference_names[i],txt_scores=txt_scores,img_scores=img_scores)
        file_utils.write_a_pseudo_target_file(f'{dataset_path}/task/{task}/pseudo_targets/{Check_LLM_model_name}_blip2_t5_{model_name}_pseudo_targets_loop_{loop}_{task}.json',reference_name=reference_names[i],rank1=rank1,rank2=rank2,pseudo_target1=pseudo_target1, confidence1=confidence1, pseudo_target2=pseudo_target2, confidence2=confidence2)
        
        pseudo_targets1.append(pseudo_target1)
        pseudo_targets2.append(pseudo_target2)
        confidences1.append(confidence1)
        confidences2.append(confidence2)
        ranks1.append(rank1)
        ranks2.append(rank2)

        if confidence1 > threshold:
            txt_check_index[i] = False
        if confidence2 > threshold:
            img_check_index[i] = False
        # print(f"Sample {i}:")
        # print(f"  Text path - Pseudo target: {pseudo_target1}, Confidence: {confidence1:.4f}")
        # print(f"  Image path - Pseudo target: {pseudo_target2}, Confidence: {confidence2:.4f}")

    return candidates1, candidates2, ranks1,ranks2,pseudo_targets1, confidences1, pseudo_targets2, confidences2, txt_check_index, img_check_index

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()


def text_encoding(device, clip_model, input_captions, batch_size=32, mode='default'):
    n_iter = int(np.ceil(len(input_captions)/batch_size))
    predicted_features = []
        
    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
        
        if hasattr(clip_model, 'tokenizer'):
            tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
        else:
            tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)

        clip_text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)        
        
    return torch.nn.functional.normalize(predicted_features, dim=-1)
    

prompt_ensemble = [
    'A bad photo of a {}',
    'A photo of many {}',
    'A sculpture of a {}',
    'A photo of the hard to see {}',
    'A low resolution photo of the {}',
    'A rendering of a {}',
    'Graffiti of a {}',
    'A bad photo of the {}',
    'A cropped photo of the {}',
    'A tattoo of a {}',
    'The embroidered {}',
    'A photo of a hard to see {}',
    'A bright photo of a {}',
    'A photo of a clean {}',
    'A photo of a dirty {}',
    'A dark photo of the {}',
    'A drawing of a {}',
    'A photo of my {}',
    'The plastic {}',
    'A photo of the cool {}',
    'A close-up photo of a {}',
    'A black and white photo of the {}',
    'A painting of the {}',
    'A painting of a {}',
    'A pixelated photo of the {}',
    'A sculpture of the {}',
    'A bright photo of the {}',
    'A cropped photo of a {}',
    'A plastic {}',
    'A photo of the dirty {}',
    'A jpeg corrupted photo of a {}',
    'A blurry photo of the {}',
    'A photo of the {}',
    'A good photo of the {}',
    'A rendering of the {}',
    'A {} in a video game',
    'A photo of one {}',
    'A doodle of a {}',
    'A close-up photo of the {}',
    'A photo of a {}',
    'The origami {}',
    'The {} in a video game',
    'A sketch of a {}',
    'A doodle of the {}',
    'A origami {}',
    'A low resolution photo of a {}',
    'The toy {}',
    'A rendition of the {}',
    'A photo of the clean {}',
    'A photo of a large {}',
    'A rendition of a {}',
    'A photo of a nice {}',
    'A photo of a weird {}',
    'A blurry photo of a {}',
    'A cartoon {}',
    'Art of a {}',
    'A sketch of the {}',
    'A embroidered {}',
    'A pixelated photo of a {}',
    'Itap of the {}',
    'A jpeg corrupted photo of the {}',
    'A good photo of a {}',
    'A plushie {}',
    'A photo of the nice {}',
    'A photo of the small {}',
    'A photo of the weird {}',
    'The cartoon {}',
    'Art of the {}',
    'A drawing of the {}',
    'A photo of the large {}',
    'A black and white photo of a {}',
    'The plushie {}',
    'A dark photo of a {}',
    'Itap of a {}',
    'Graffiti of the {}',
    'A toy {}',
    'Itap of my {}',
    'A photo of a cool {}',
    'A photo of a small {}',
    'A tattoo of the {}',
]
