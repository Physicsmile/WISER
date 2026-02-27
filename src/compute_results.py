import json
import os
from typing import List, Dict, Union

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm
import datetime

@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,dataset_name,dataset_path,task,
    split: str='val',
    loop:str = 0,
    ways: str="",
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    # Move the features to the device
    index_features = torch.nn.functional.normalize(index_features).to(device)
    predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    print("index_features.shape:",index_features.shape)
    print("predicted_features.shape:",predicted_features.shape)
    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    output_metrics = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    json.dump(output_metrics,open(f'{dataset_path}/task/{task}/result_{dataset_name}_{ways}_loop_{loop}_{get_time()}.json','w'),indent=6)
    return output_metrics,sorted_index_names


@torch.no_grad()
def fiq_fuse2paths(
    device: torch.device,
    predicted_img_features: torch.Tensor,
    predicted_txt_features: torch.Tensor,
    candidates1: list,
    candidates2: list,
    target_names: List,
    reference_names:list,
    index_features: torch.Tensor,
    index_names: List,dataset_name,dataset_path,task,
    split: str='val',
    loop:str = 0,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set given the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    topk = 50  
    top2k = 2 * topk  
    
    Q = len(target_names)  
    
    recall_1_list = []
    recall_5_list = []
    recall_10_list = []
    recall_50_list = []
    all_sorted_index_names = []
    recall_results = []

    for i in range(Q):
        query_candidates1 = candidates1[i] 
        query_candidates2 = candidates2[i]  
        target_name = target_names[i]  
        
        candidate_set = set()
        
        for candidate in query_candidates1:
            candidate_path = candidate[1]  
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            candidate_set.add(candidate_name)
        
        for candidate in query_candidates2:
            candidate_path = candidate[1] 
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            candidate_set.add(candidate_name)
        
        candidate_list = list(candidate_set)
        assert len(candidate_list) <= top2k
        
        candidate_confidence_sum = {}
        
        confidence_map1 = {}
        for candidate in query_candidates1:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            confidence = candidate[2]  
            confidence_map1[candidate_name] = confidence
        
        confidence_map2 = {}
        for candidate in query_candidates2:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            confidence = candidate[2]  
            confidence_map2[candidate_name] = confidence
        
        for candidate_name in candidate_list:
            conf1 = confidence_map1.get(candidate_name, 0.0)
            conf2 = confidence_map2.get(candidate_name, 0.0)
            candidate_confidence_sum[candidate_name] = conf1 + conf2
        
        sorted_candidates = sorted(
        candidate_confidence_sum.items(),
        key=lambda x: (
            -(confidence_map1.get(x[0], 0.0) + confidence_map2.get(x[0], 0.0)),
            -max(confidence_map1.get(x[0], 0.0), confidence_map2.get(x[0], 0.0)),
            -confidence_map1.get(x[0], 0.0),
            x[0]
        )
    )

        sorted_index_names_query = [candidate_name for candidate_name, _ in sorted_candidates]
        all_sorted_index_names.append(sorted_index_names_query)
        
        target_position = None
        for pos, candidate_name in enumerate(sorted_index_names_query):
            if candidate_name == target_name:
                target_position = pos
                break
        
        if target_position is not None:
            recall_1 = 1.0 if target_position < 1 else 0.0
            recall_5 = 1.0 if target_position < 5 else 0.0
            recall_10 = 1.0 if target_position < 10 else 0.0
            recall_50 = 1.0 if target_position < 50 else 0.0
        else:
            recall_1 = 0.0
            recall_5 = 0.0
            recall_10 = 0.0
            recall_50 = 0.0

        t2i_cand = sorted(confidence_map1.items(), key=lambda x: x[1], reverse=True)
        t2i_cand_sorted = [candidate_name for candidate_name, _ in t2i_cand]
        i2i_cand = sorted(confidence_map2.items(), key=lambda x: x[1], reverse=True)
        i2i_cand_sorted = [candidate_name for candidate_name, _ in i2i_cand]

        recall_1_list.append(recall_1)
        recall_5_list.append(recall_5)
        recall_10_list.append(recall_10)
        recall_50_list.append(recall_50)
        recall_result = {
            "image_index": reference_names[i],
            "target_name": target_name,
            "top_names": sorted_index_names_query[:50],  
            "t2i_top_names": t2i_cand_sorted[:50],
            "i2i_top_names": i2i_cand_sorted[:50]
        }
        recall_results.append(recall_result)    

    output_metrics = {
        'Recall@1': (sum(recall_1_list) / Q) * 100,
        'Recall@5': (sum(recall_5_list) / Q) * 100,
        'Recall@10': (sum(recall_10_list) / Q) * 100,
        'Recall@50': (sum(recall_50_list) / Q) * 100
    }
    recall_results_filename = f"{dataset_path}/task/{task}/top_rank_fused_loop_{loop}_{task}_{get_time()}.json"
    with open(recall_results_filename, 'w') as f:
        json.dump(recall_results, f, indent=4)
    
    print(f"Resutls saved to : {recall_results_filename}")

    json.dump(output_metrics,open(f'{dataset_path}/task/{task}/result_{dataset_name}_loop_{loop}_{get_time()}.json','w'),indent=6)
    return output_metrics, all_sorted_index_names


@torch.no_grad()
def cirr(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],dataset_name,dataset_path,task,
    preload_dict: Dict[str, Union[str, None]]=[],
    split: str='val',  
    loop:str = 0,
    ways: str="",
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Computes Recall@1, 5, 10 and 50. If given a test set, will generate submittable file.
    """   
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    distances = 1 - predicted_features @ index_features.T
    if distances.ndim == 3:
        distances = distances.mean(dim=1)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    resize = len(sorted_index_names) if split == 'test' else len(target_names)
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(resize, -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    if split == 'test':
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
        for pair_id, prediction in zip(query_ids, sorted_index_names):
            pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
        for pair_id, prediction in zip(query_ids, sorted_group_names):
            pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)
    
        submissions_folder_path = f'{dataset_path}/task/{task}/test_submissions_{dataset_name}_{ways}_cirr_loop_{loop}_{get_time()}.json'
        group_submissions_folder_path = f'{dataset_path}/task/{task}/subset_test_submissions_{dataset_name}_{ways}_cirr_loop_{loop}_{get_time()}.json'

        with open(submissions_folder_path, 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(group_submissions_folder_path, 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        return None,sorted_index_names
            
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))    
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)
    labels_sum = torch.sum(labels, dim=-1).int()
    expected_ones = torch.ones(len(target_names)).int()

    output_metrics = {f'recall@{key}': (torch.sum(labels[:, :key]) / len(labels)).item() * 100 for key in [1, 5, 10, 50]}
    output_metrics.update({f'group_recall@{key}': (torch.sum(group_labels[:, :key]) / len(group_labels)).item() * 100 for key in [1, 2, 3]})
    
    return output_metrics,sorted_index_names


def cirr_fuse2paths(
    device: torch.device,
    predicted_img_features: torch.Tensor,
    predicted_txt_features: torch.Tensor,
    candidates1: list,
    candidates2: list,
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],dataset_name,dataset_path,task,
    preload_dict: Dict[str, Union[str, None]]=[],
    split: str='val',  
    loop:str = 0,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Computes Recall@1, 5, 10 and 50. If given a test set, will generate submittable file.
    """   
    topk = 50  
    top2k = 2 * topk  
    Q = len(reference_names)  

    all_sorted_index_names = []
    all_results = []
    group_results = []

    all_labels = []
    all_group_labels = []

    all_sorted_index_names = []
    recall_1_list = []
    recall_5_list = []
    recall_10_list = []
    recall_50_list = []
    group_recall_1_list = []
    group_recall_2_list = []
    group_recall_3_list = []
    
    recall_results = []

    for i in range(Q):
        query_candidates1 = candidates1[i]  
        query_candidates2 = candidates2[i]  
        target_name = target_names[i] if split != 'test' else None
        reference_name = reference_names[i]
        query_id = query_ids[i] if split == 'test' else None
        
        candidate_set = set()
        
        for candidate in query_candidates1:
            candidate_path = candidate[1]  
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            candidate_set.add(candidate_name)
        
        for candidate in query_candidates2:
            candidate_path = candidate[1] 
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            candidate_set.add(candidate_name)
        
        candidate_list = list(candidate_set)
        assert len(candidate_list) <= top2k
        
        candidate_confidence_sum = {}
        
        confidence_map1 = {}
        for candidate in query_candidates1:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            confidence = candidate[2] 
            confidence_map1[candidate_name] = confidence
        
        confidence_map2 = {}
        for candidate in query_candidates2:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            confidence = candidate[2] 
            confidence_map2[candidate_name] = confidence
        
        for candidate_name in candidate_list:
            conf1 = confidence_map1.get(candidate_name, 0.0)
            conf2 = confidence_map2.get(candidate_name, 0.0)
            candidate_confidence_sum[candidate_name] = conf1 + conf2
        
        sorted_candidates = sorted(
            candidate_confidence_sum.items(),
            key=lambda x: (
                -(confidence_map1.get(x[0], 0.0) + confidence_map2.get(x[0], 0.0)),
                -max(confidence_map1.get(x[0], 0.0), confidence_map2.get(x[0], 0.0)),
                -confidence_map1.get(x[0], 0.0),
                x[0]
            )
        )

        sorted_index_name = [candidate_name for candidate_name, _ in sorted_candidates]
        filtered_sorted_names = [name for name in sorted_index_name if name != reference_name]
        
        if split == 'test':
            top50_results = filtered_sorted_names[:50] if len(filtered_sorted_names) >= 50 else filtered_sorted_names
            all_results.append((query_id, top50_results))
            
            current_targets = targets[i] 
            group_candidates = [name for name in filtered_sorted_names if name in current_targets]
            top3_group_results = group_candidates[:3] if len(group_candidates) >= 3 else group_candidates
            group_results.append((query_id, top3_group_results))
        else:
            current_target_name = target_names[i]
            current_group_targets = targets[i]
            
            recall_1 = 1 if current_target_name in filtered_sorted_names[:1] else 0
            recall_5 = 1 if current_target_name in filtered_sorted_names[:5] else 0
            recall_10 = 1 if current_target_name in filtered_sorted_names[:10] else 0
            recall_50 = 1 if current_target_name in filtered_sorted_names[:50] else 0
            
            group_candidates_in_top = [name for name in filtered_sorted_names[:3] if name in current_group_targets]
            group_recall_1 = 1 if len(group_candidates_in_top[:1]) > 0 else 0
            group_recall_2 = 1 if len(group_candidates_in_top[:2]) > 0 else 0
            group_recall_3 = 1 if len(group_candidates_in_top[:3]) > 0 else 0
            
            t2i_cand = sorted(confidence_map1.items(), key=lambda x: x[1], reverse=True)
            t2i_cand_sorted = [candidate_name for candidate_name, _ in t2i_cand]
            i2i_cand = sorted(confidence_map2.items(), key=lambda x: x[1], reverse=True)
            i2i_cand_sorted = [candidate_name for candidate_name, _ in i2i_cand]

            recall_1_list.append(recall_1)
            recall_5_list.append(recall_5)
            recall_10_list.append(recall_10)
            recall_50_list.append(recall_50)
            group_recall_1_list.append(group_recall_1)
            group_recall_2_list.append(group_recall_2)
            group_recall_3_list.append(group_recall_3)
            
            recall_result = {
                "image_index": reference_name,
                "target_name": target_name,
                "top_names": filtered_sorted_names[:50],
                "t2i_top_names": t2i_cand_sorted[:50],
                "i2i_top_names": i2i_cand_sorted[:50]
            }
            recall_results.append(recall_result)    
            
            all_sorted_index_names.append(filtered_sorted_names)

    if split == 'test':
        pairid_to_retrieved_images = {}
        pairid_to_group_retrieved_images = {}
        
        for query_id, prediction in all_results:
            pairid_to_retrieved_images[str(int(query_id))] = prediction
        
        for query_id, prediction in group_results:
            pairid_to_group_retrieved_images[str(int(query_id))] = prediction

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)

        submissions_folder_path = f'{dataset_path}/task/{task}/test_submissions_{dataset_name}_cirr_loop_{loop}_{get_time()}.json'
        group_submissions_folder_path = f'{dataset_path}/task/{task}/subset_test_submissions_{dataset_name}_cirr_loop_{loop}_{get_time()}.json'
        
        print(submissions_folder_path)

        with open(submissions_folder_path, 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(group_submissions_folder_path, 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        
        return None, all_results
    
    else:
        output_metrics = {
            'recall@1': (sum(recall_1_list) / Q) * 100,
            'recall@5': (sum(recall_5_list) / Q) * 100,
            'recall@10': (sum(recall_10_list) / Q) * 100,
            'recall@50': (sum(recall_50_list) / Q) * 100,
            'group_recall@1': (sum(group_recall_1_list) / Q) * 100,
            'group_recall@2': (sum(group_recall_2_list) / Q) * 100,
            'group_recall@3': (sum(group_recall_3_list) / Q) * 100
        }
        
        recall_results_filename = f"{dataset_path}/task/{task}/top_rank_fused_loop_{loop}_{task}_{get_time()}.json"
        with open(recall_results_filename, 'w') as f:
            json.dump(recall_results, f, indent=4)
        
        print(f"Results saved to: {recall_results_filename}")

        with open(f'{dataset_path}/task/{task}/result_{dataset_name}_loop_{loop}_{get_time()}.json','w') as f:
            json.dump(output_metrics, f, indent=6)
            
        return output_metrics, all_sorted_index_names

@torch.no_grad()
def circo(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List,
    query_ids: Union[np.ndarray,List],dataset_name,dataset_path,task,
    preload_dict: Dict[str, Union[str, None]]=[],
    split: str='val',
    loop:str = 0,
    ways: str="",
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names.
    Computes mAP@5, 10, 25 and 50. If test-split, generates submittable file.
    """
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)    
    if split == 'test':
        print('Generating test submission file!')
        similarity = predicted_features @ index_features.T
        if similarity.ndim == 3:
            similarity = similarity.mean(dim=1)                    
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]

        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        json.dump(queryid_to_retrieved_images,open(f'{dataset_path}/task/{task}/test_submissions_{dataset_name}_{ways}_loop_{loop}_{get_time()}.json','w'),indent=6)
 
        return None,sorted_index_names
    
    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
    sorted_index_names_list = []
    for predicted_feature, target_name, sub_targets in tqdm.tqdm(zip(predicted_features, target_names, targets), total=len(predicted_features), desc='Computing Metric.'):
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']  
        similarity = predicted_feature @ index_features.T
        if similarity.ndim == 2:

            similarity = similarity.mean(dim=0)
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        sorted_index_names_list.append(sorted_index_names)
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        assert target_name == sub_targets[0], f"Target name not in GTs {target_name} {sub_targets}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        
        for key in retrievals:
            recalls[key].append(float(torch.sum(single_gt_labels[:key])))
    sorted_index_names_list = np.array(sorted_index_names_list)
    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}
    output_metrics.update({f'recall@{key}': np.mean(item) * 100 for key, item in recalls.items()})
    return output_metrics,sorted_index_names_list


@torch.no_grad()
def circo_fuse2paths(
    device: torch.device,
    predicted_img_features: torch.Tensor,
    predicted_txt_features: torch.Tensor,
    candidates1: list,
    candidates2: list,
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],dataset_name,dataset_path,task,
    preload_dict: Dict[str, Union[str, None]]=[],
    split: str='val',  
    loop:str = 0,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names.
    Computes mAP@5, 10, 25 and 50. If test-split, generates submittable file.
    """
    if split == "test":
        topk = 50  
        top2k = 2 * topk  
        Q = len(reference_names)  

        all_sorted_index_names = []
        all_results = []
        group_results = []
        sorted_index_names = []
        for i in range(Q):
            query_candidates1 = candidates1[i]  
            query_candidates2 = candidates2[i]  
            target_name = target_names[i] if split != 'test' else None
            reference_name = reference_names[i]
            query_id = query_ids[i] if split == 'test' else None
            
            candidate_set = set()
            
            for candidate in query_candidates1:
                candidate_path = candidate[1]  
                candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
                candidate_set.add(candidate_name)
            
            for candidate in query_candidates2:
                candidate_path = candidate[1] 
                candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
                candidate_set.add(candidate_name)
            
            candidate_list = list(candidate_set)
            assert len(candidate_list) <= top2k
            
            candidate_confidence_sum = {}
            
            confidence_map1 = {}
            for candidate in query_candidates1:
                candidate_path = candidate[1]
                candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
                confidence = candidate[2]  
                confidence_map1[candidate_name] = confidence
            
            confidence_map2 = {}
            for candidate in query_candidates2:
                candidate_path = candidate[1]
                candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
                confidence = candidate[2] 
                confidence_map2[candidate_name] = confidence
            
            for candidate_name in candidate_list:
                conf1 = confidence_map1.get(candidate_name, 0.0)
                conf2 = confidence_map2.get(candidate_name, 0.0)
                candidate_confidence_sum[candidate_name] = conf1 + conf2
            
            sorted_candidates = sorted(
                candidate_confidence_sum.items(),
                key=lambda x: (
                    -(confidence_map1.get(x[0], 0.0) + confidence_map2.get(x[0], 0.0)),
                    -max(confidence_map1.get(x[0], 0.0), confidence_map2.get(x[0], 0.0)),
                    -confidence_map1.get(x[0], 0.0),
                    x[0]
                )
            )

            sorted_index_name = [candidate_name for candidate_name, _ in sorted_candidates]
            sorted_index_names.append(sorted_index_name)

        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50] for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        output_path = f'{dataset_path}/task/{task}/test_submissions_{dataset_name}_loop_{loop}_{get_time()}.json'
        json.dump(queryid_to_retrieved_images,open(output_path,'w'),indent=6)
     
        return None,sorted_index_names

    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
    sorted_index_names_list = []
    
    recall_results = []
    
    Q = len(reference_names)
    for i in range(Q):
        query_candidates1 = candidates1[i]
        query_candidates2 = candidates2[i]
        target_name = target_names[i]
        reference_name = reference_names[i]
        sub_targets = targets[i]
        
        candidate_set = set()
        for candidate in query_candidates1:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            candidate_set.add(candidate_name)
        for candidate in query_candidates2:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            candidate_set.add(candidate_name)
        
        candidate_list = list(candidate_set)
        
        confidence_map1 = {}
        for candidate in query_candidates1:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            confidence = candidate[2]
            confidence_map1[candidate_name] = confidence
        
        confidence_map2 = {}
        for candidate in query_candidates2:
            candidate_path = candidate[1]
            candidate_name = os.path.splitext(os.path.basename(candidate_path))[0]
            confidence = candidate[2]
            confidence_map2[candidate_name] = confidence
        
        candidate_confidence_sum = {}
        for candidate_name in candidate_list:
            conf1 = confidence_map1.get(candidate_name, 0.0)
            conf2 = confidence_map2.get(candidate_name, 0.0)
            candidate_confidence_sum[candidate_name] = conf1 + conf2
        
        sorted_candidates = sorted(
            candidate_confidence_sum.items(),
            key=lambda x: (
                -(confidence_map1.get(x[0], 0.0) + confidence_map2.get(x[0], 0.0)),
                -max(confidence_map1.get(x[0], 0.0), confidence_map2.get(x[0], 0.0)),
                -confidence_map1.get(x[0], 0.0),
                x[0]
            )
        )

        sorted_index_names = [candidate_name for candidate_name, _ in sorted_candidates]
        sorted_index_names_list.append(sorted_index_names)
        
        t2i_cand = sorted(confidence_map1.items(), key=lambda x: x[1], reverse=True)
        t2i_cand_sorted = [candidate_name for candidate_name, _ in t2i_cand]
        i2i_cand = sorted(confidence_map2.items(), key=lambda x: x[1], reverse=True)
        i2i_cand_sorted = [candidate_name for candidate_name, _ in i2i_cand]
        
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        recall_result = {
            "image_index": reference_name,
            "target_name": target_name,
            "top_names": sorted_index_names[:50],
            "t2i_top_names": t2i_cand_sorted[:50],
            "i2i_top_names": i2i_cand_sorted[:50]
        }
        recall_results.append(recall_result)
    
    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}

    recall_results_filename = f"{dataset_path}/task/{task}/top_rank_fused_loop_{loop}_{task}_{get_time()}.json"
    with open(recall_results_filename, 'w') as f:
        json.dump(recall_results, f, indent=4)
    print(f"Results saved to: {recall_results_filename}")

    with open(f'{dataset_path}/task/{task}/top_rank_fused_{dataset_name}_loop_{loop}_{get_time()}.json','w') as f:
        json.dump(output_metrics, f, indent=6)
        
    return output_metrics,sorted_index_names_list


@torch.no_grad()
def genecis(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    index_features: torch.Tensor, 
    index_ranks: List,
    topk: List[int] = [1, 2, 3],
    loop:str = 0,
    **kwargs    
) -> Dict[str, float]:
    
    predicted_features = torch.nn.functional.normalize(predicted_features.float(), dim=-1).to(device)
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1).to(device)
    
    if predicted_features.ndim == 3:
        similarities = predicted_features.bmm(index_features.permute(0,2,1)).mean(dim=1)
    else:
        similarities = (predicted_features[:, None, :] * index_features).sum(dim=-1)

    # # Sort the similarities in ascending order (closest example is the predicted sample)
    _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

    # Compute recall at K
    if isinstance(index_ranks, list):
        index_ranks = torch.stack(index_ranks)
    index_ranks = index_ranks.to(device)
    
    output_metrics = {f'R@{k}': get_recall(sort_idxs[:, :k], index_ranks) * 100 for k in topk}

    return output_metrics

def get_time()->str:
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d-%H_%M_%S")
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
    
