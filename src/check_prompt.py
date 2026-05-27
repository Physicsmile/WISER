import json
import requests
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum, auto
import math
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import os
from tqdm import tqdm
from typing import List, Dict, Any
from qwen_vl_utils import process_vision_info

class CheckModel(Enum):
    gpt_4o = auto()
    gpt_4o_mini = auto()
    gpt_3_5_turbo = auto()
    qwen_turbo = auto()
    qwen2_5_vl_7b_instruct = auto()  
    gemini_2_5_pro = auto()
    deepseek_r1 = auto()
    @staticmethod
    def from_string(s: str):
        try:
            return CheckModel[s.replace("-", "_").lower()]
        except KeyError:
            raise ValueError(f"Unknown model type: {s}")
        
class ModelHandler:
    def __init__(self, model_type: str, device: torch.device, openai_key: str = None):
        self.model_type = CheckModel.from_string(model_type.replace("-", "_"))
        self.device = device
        self.openai_key = openai_key

        # Load model depending on whether it's local or API-based
        if self.model_type in [CheckModel.gpt_4o_mini, CheckModel.gpt_4o, CheckModel.gpt_3_5_turbo, CheckModel.qwen_turbo, CheckModel.gemini_2_5_pro, CheckModel.gemini_3_pro_preview,CheckModel.deepseek_r1]:
            self.model = None
            self.tokenizer = None
            self.chat_function = self._chat_api
        else:
            self.model, self.tokenizer = self._load_model_and_process(device)
            self.chat_function = self._chat_local


    def _load_model_and_process(self, device: torch.device):
        tokenizer = AutoTokenizer.from_pretrained(self.model_type.value)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_type.value, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device, attn_implementation="flash_attention_2"
        )
        return model, tokenizer

    def load_image(self, image_path, max_size=1024):
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (224, 224), color='white')
    
    def _chat_local(self, image_caption, relative_caption,txt_top_captions, img_top_captions, device, max_length=800):
        prompt = self._generate_check_prompt(
            image_caption=image_caption,
            relative_caption=relative_caption,
            txt_top_caption=txt_top_captions,
            img_top_caption=img_top_captions
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        generated_ids = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=1.0,
            num_beams=5,  
            early_stopping=True
        )
        
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return output_text

    
    def _load_model_and_process(self, device: torch.device):
        if self.model_type == CheckModel.qwen2_5_vl_7b_instruct:
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                min_pixels=4*28*28,
                max_pixels=2048*28*28
            )
            processor.tokenizer.padding_side = 'left'
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.float16,
                device_map={"": device},
                attn_implementation="sdpa" # "flash_attention_2"
            )
            model.eval()
            return model, processor
        else:
            assert "Need to Be Finished"
            pass
    
    def _send_request(self, url, headers, payload, openai_key, max_retries=5, backoff=0.25):
        """Generic function for sending requests with retry logic."""
        for attempt in range(max_retries):
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url=url,
                    api_key=openai_key
                )
                completion = client.chat.completions.create(
                    model=payload["model"],
                    messages=payload['messages']
                )
                
                if not completion.choices:
                    print(f"Attempt {attempt + 1}: No choices in response")
                    continue  
                    
                if not completion.choices[0].message.content:
                    print(f"Attempt {attempt + 1}: Empty message content")
                    continue  
                    
                return completion.choices[0].message.content
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                
            time.sleep(backoff)
            backoff *= 2  
        
        return "Request failed after multiple attempts"

    def _chat_api(self, prompt_type, image_caption, relative_caption,txt_top_captions, img_top_captions,device, max_length=800):
        model_map = {
            CheckModel.gpt_4o: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o"
            },
            CheckModel.gpt_4o_mini: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o-mini"
            },
            CheckModel.gpt_3_5_turbo: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-3.5-turbo"
            },
            CheckModel.qwen_turbo: {
                "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model_name": "qwen-plus"
            },
            CheckModel.gemini_2_5_pro: {
                "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "model_name": "gemini-2.5-pro"
            },
            CheckModel.deepseek_r1: {
                "url": "https://api.deepseek.com/v1",
                "model_name": "DeepSeek-R1"
            },

        }

        if self.model_type not in model_map:
            raise ValueError("Unsupported API model.")

        model_info = model_map[self.model_type]
        if prompt_type == "t2i":
            prompt = self._generate_t2i_check_prompt(image_caption, relative_caption, txt_top_captions)
        elif prompt_type == "i2i":
            prompt = self._generate_i2i_check_prompt(image_caption, relative_caption, img_top_captions)
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': f'Bearer {self.openai_key}'
        }
        # print("Prompt:", prompt)
        payload = {
            "model": model_info["model_name"],
            "messages": [{"role": "user", "content": prompt}]
        }

        return self._send_request(model_info["url"], headers, payload, self.openai_key)
    
    
    def _generate_t2i_check_prompt(self, image_caption, relative_caption, txt_top_caption):
        return f'''
        Assume you are an experienced composed image retrieval expert, skilled at precisely generating new image descriptions based on a reference image's description and the user's modification instructions.
        You excel at creating modified descriptions that can retrieve images matching the user's requested changes through vector retrieval.
        Your task is to help improve the effectiveness of compositional image retrieval by generating precise modification suggestions that will assist another large language model (LLM) in producing a better image description.
        Please note that this LLM has received the reference image's description and the user's modification instructions, and already generated a modified description.
        Moreover, a retrieval has been performed based on this modified description. Thus your task is to analyze the last retrieval result and provide modification suggestions and please follow the below steps to finish this task.
        
        Step 1: Identifying Modifications
        Your first task is to identify the modifications and generate corresponding modification phrases.
        Specifically, here is the description of the reference image: "{image_caption}." Here are the user's modification requests: "{relative_caption}"
        By deeply understanding the image description and the user's modifications, please generate the following two types of modification phrases:
        1. If the modification involves changing the characteristics of an entity in the original reference image, please specify the changes,
        2. If the modification involves adding or deleting an entity, please specify the additions or deletions.
        Please note that the user's modifications may lack a subject; in such cases, infer and supply the object corresponding to the modification.
        Only include modifications explicitly mentioned by the user. If a certain type of modification is not present, you do not need to provide it and should avoid generating unspecified content.

        Step 2: Analyzing the Retrieved Image 
        Compare the modification phrases identified in Step 1 with the description of the retrieved image : "{txt_top_caption}". Note that this retrieval is performed with the modified description generated by another LLM, which has been mentioned above.
        Determine if the retrieved image meets the user's modification instructions.
        If it matches after excluding subjective modifications (e.g., "casual," "relaxed"), respond with: "Good retrieval, no more loops needed."
        If there are unmet modification phrases, proceed to Step 3.

        Step 3: Providing Modification Suggestions 
        For any unmet modifications identified in Step 2, suggest targeted changes to help the LLM regenerate an improved modified description. Keep suggestions concise and specific to ensure they effectively guide the LLM.
        
        **Output format:**
        "Suggestion: <concise, actionable suggestion in 10-20 words>"
        '''


    def _generate_i2i_check_prompt(self, image_caption, relative_caption, img_top_caption):
        return f'''
        Assume you are an experienced composed image retrieval expert, skilled at precisely generating new image based on a reference image's description and the user's modification instructions.
        You excel at creating modified images that can retrieve images matching the user's requested changes through vector retrieval.
        Your task is to help improve the effectiveness of compositional image retrieval by generating precise modification suggestions that will assist another multimodal large language model (MLLM) in producing a better image.
        Please note that this MLLM has received the reference image's description and the user's modification instructions, and already generated a modified image.
        Moreover, a retrieval has been performed based on this modified image. Thus your task is to analyze the last retrieval result and provide modification suggestions and please follow the below steps to finish this task.
        
        Step 1: Identifying Modifications
        Your first task is to identify the modifications and generate corresponding modification phrases.
        Specifically, here is the description of the reference image: "{image_caption}." Here are the user's modification requests: "{relative_caption}"
        By deeply understanding the image description and the user's modifications, please generate the following two types of modification phrases:
        1. If the modification involves changing the characteristics of an entity in the original reference image, please specify the changes,
        2. If the modification involves adding or deleting an entity, please specify the additions or deletions.
        Please note that the user's modifications may lack a subject; in such cases, infer and supply the object corresponding to the modification.
        Only include modifications explicitly mentioned by the user. If a certain type of modification is not present, you do not need to provide it and should avoid generating unspecified content.

        Step 2: Analyzing the Retrieved Image 
        Compare the modification phrases identified in Step 1 with the description of the retrieved image : "{img_top_caption}". Note that this retrieval is performed with the modified image generated by another MLLM, which has been mentioned above.
        Determine if the retrieved image meets the user's modification instructions.
        If it matches after excluding subjective modifications (e.g., "casual," "relaxed"), respond with: "Good retrieval, no more loops needed."
        If there are unmet modification phrases, proceed to Step 3.

        Step 3: Providing Modification Suggestions 
        For any unmet modifications identified in Step 2, suggest targeted changes to help the MLLM regenerate an improved modified image. Keep suggestions concise and specific to ensure they effectively guide the MLLM.
        
        **Output format:**
        "Suggestion: <concise, actionable suggestion in 10-20 words>"
        '''