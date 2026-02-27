from argparse import Namespace
import torch
import os
import data_utils
import utils
from datasets import FashionIQDataset,CIRRDataset,CIRCODataset
import compute_results
import clip
from transformers import CLIPProcessor, CLIPModel
import termcolor
from classes import load_clip_model_and_preprocess
import wandb
from bagel_inference import BagelImageEditor


import file_utils

class Experiment:
    def __init__(self,args:Namespace) -> None:
        super().__init__()
        self.args = args
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)
        self.device = torch.device(f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu')
        
    def run(self):
        clip_model,clip_processor = self.load_Clip_model()
        bagel_editor = self.load_Bagel_model()
        target_datasets,query_datasets,compute_results_function,compute_results_fuse2paths_function,pairings = self.load_dataset(clip_processor)
        self.evaluate(bagel_editor,query_datasets, target_datasets, pairings,compute_results_function,compute_results_fuse2paths_function,clip_model,clip_processor)
        print("Finish.")
        return
    
    def data_preprocessing(self):
        return
    def load_Clip_model(self):
        clip_device = self.device
        clip_model,clip_processor = load_clip_model_and_preprocess(dataset_path=self.dataset_path,clip_type=self.clip,device=clip_device)
        print('Done.')
        return clip_model,clip_processor
    
    def load_Bagel_model(self):
        bagel_editor = BagelImageEditor(self.bagel_model_path)
        return bagel_editor
    
    def load_dataset(self,clip_processor:CLIPProcessor):
        ### Load Evaluation Datasets.
        target_datasets, query_datasets, pairings = [], [], []
        if 'fashioniq' in self.dataset.lower():
            dress_type = self.dataset.split('_')[-1]
            target_datasets.append(FashionIQDataset(self.dataset_path, self.split, [dress_type], 'classic', clip_processor))
            query_datasets.append(FashionIQDataset(self.dataset_path, self.split, [dress_type], 'relative', clip_processor))
            pairings.append(dress_type)
            compute_results_function = compute_results.fiq
            compute_results_fuse2paths_function = compute_results.fiq_fuse2paths
        
        
        elif self.dataset.lower() == 'cirr':
            split = 'test1' if self.split == 'test' else self.split
            target_datasets.append(CIRRDataset(self.dataset_path, split, 'classic', clip_processor))
            query_datasets.append(CIRRDataset(self.dataset_path, split, 'relative', clip_processor))
            compute_results_function = compute_results.cirr
            pairings.append('default')
            compute_results_fuse2paths_function = compute_results.cirr_fuse2paths
            
            
        elif self.dataset.lower() == 'circo':
            target_datasets.append(CIRCODataset(self.dataset_path, self.split, 'classic', clip_processor))
            query_datasets.append(CIRCODataset(self.dataset_path, self.split, 'relative', clip_processor))
            compute_results_function = compute_results.circo
            pairings.append('default')
            compute_results_fuse2paths_function = compute_results.circo_fuse2paths

        return target_datasets,query_datasets,compute_results_function,compute_results_fuse2paths_function,pairings

    def evaluate(self,bagel_editor,query_datasets, target_datasets, pairings,compute_results_function,compute_results_fuse2paths_function,clip_model,clip_processor):
        preload_dict = {key: None for key in ['img_features','new_captions', 'captions', 'img_paths','mods','suggestions','pseudo_targets','pseudo_targets_loop2']}
        file_utils.init_folder(self.dataset_path, self.task)
        if 'mods' in self.preload:
            preload_dict['mods'] = f'{self.dataset_path}/task/{self.task}/modified_captions/{self.preload_modified_captions_file}'
        if 'captions' in self.preload:
            preload_dict['captions'] = f'{self.dataset_path}/preload/image_captions/{self.preload_image_captions_file}'
        if 'img_paths' in self.preload:
            preload_dict['img_paths'] = f'{self.dataset_path}/preload/image_paths/{self.preload_image_paths_file}'
        if 'pseudo_targets' in self.preload:
            preload_dict['pseudo_targets'] = f'{self.dataset_path}/task/{self.task}/pseudo_targets/{self.preload_pseudo_targets}'
        if 'candidates' in self.preload:
            preload_dict['candidates'] = f'{self.dataset_path}/task/{self.task}/pseudo_targets/{self.preload_candidates}'
        if 'suggestions' in self.preload:
            preload_dict['suggestions'] = f'{self.dataset_path}/task/{self.task}/suggestions/{self.preload_suggestions}'
            print("preload_dict['suggestions']:",preload_dict['suggestions'])
        if 'img_features' in self.preload:
            preload_dict['img_features'] = f'{self.dataset_path}/preload/img_features/{self.clip}_{self.dataset}_{self.split}.pkl'
        if 'edited_images' in self.preload:
            preload_dict['edit_images'] = f'{self.dataset_path}/preload/edited_images/{self.preload_edited_images_file}'
        if 'new_captions' in self.preload:
            preload_dict['new_captions'] = f'{self.dataset_path}/task/{self.task}/new_captions/{self.preload_new_captions}'
       
        for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
            termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])
            
            ### General Input Arguments.
            input_kwargs = {
                'args': self.args,
                'bagel_editor':bagel_editor,
                'dataset_name':self.dataset,'llm_prompt_args': self.llm_prompt,'retrieval':self.retrieval,
                'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model, 
                'processor': clip_processor, 'device': self.device, 'split': self.split,
                'preload_dict': preload_dict,'max_check_num':self.max_check_num,
                'Check_LLM_model_name':self.Check_LLM_model_name,'dataset_path':self.dataset_path, 'edit_img_dir':self.edit_img_dir, 
                'compute_results_function':compute_results_function,
                'VQA_LLM_model_name': self.VQA_LLM_model_name,
                'openai_key':self.openai_key,"task":self.task,'preprocess':clip_processor
            }    
            
            ### Compute Target Image Features
            print(f'Extracting target image features using CLIP: {self.clip}.')
            index_features, index_names, index_ranks, aux_data = utils.extract_image_features(
                self.device, self.dataset, target_dataset, clip_model, preload=preload_dict['img_features'])
            index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
            input_kwargs.update({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks,'LLM_model_name':self.LLM_model_name})

                
            ### Compute Method-specific Query Features.
            # This part can be interchanged with any other method implementation.
            print(f'Generating conditional query predictions (CLIP: {self.clip}.')
            out_dict = utils.generate_editimg_caption_iteration(**input_kwargs)
            input_kwargs.update(out_dict)
            
            print("Split:",input_kwargs['split'])

            result_metrics,labels = compute_results_fuse2paths_function(**input_kwargs)      
            # Print metrics.
            print('\n')
            if result_metrics is not None:
                termcolor.cprint(f'Metrics for {self.dataset.upper()} ({self.split})- {pairing}', attrs=['bold'])
                for k, v in result_metrics.items():
                    print(f"{pairing}_{k} = {v:.2f}")
            else:
                termcolor.cprint(f'No explicit metrics available for {self.dataset.upper()} ({self.split}) - {pairing}.', attrs=['bold'])            
