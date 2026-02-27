from enum import Enum, auto
import torch
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms
import clip
import os

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    

def load_clip_model_and_preprocess(dataset_path:str,clip_type:str,device:torch.device,jit:bool=False):
    print(f'Loading CLIP {clip_type}... ', end='')
    if clip_type in ['ViT-bigG-14','ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14']:
        import open_clip
        pretraining = {
            'ViT-B-32':'laion2b_s34b_b79k',
            'ViT-B-16':'laion2b_s34b_b88k',
            'ViT-L-14':'laion2b_s32b_b82k',
            'ViT-H-14':'laion2b_s32b_b79k',
            'ViT-g-14':'laion2b_s34b_b88k',
            'ViT-bigG-14':'laion2b_s39b_b160k'
        }
        weight_path = os.path.join(dataset_path, '..', 'weights', 'open_clip')
        os.makedirs(weight_path, exist_ok=True)

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_type, pretrained=pretraining[clip_type], cache_dir=weight_path)
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer(clip_type)
        clip_model.tokenizer = tokenizer
    else:
        clip_model, clip_preprocess = clip.load(clip_type, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False).to(device)
    
    return clip_model,clip_preprocess
