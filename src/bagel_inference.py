import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from inferencer import InterleaveInferencer


class BagelImageEditor:
    def __init__(self, model_path: str, max_mem_per_gpu: str = "80GiB", offload_folder: str = "/tmp/offload"):
        self.model_path = model_path
        self.max_mem_per_gpu = max_mem_per_gpu
        self.offload_folder = offload_folder
        self.model = None
        self.vae_model = None
        self.tokenizer = None
        self.inferencer = None
        self.new_token_ids = None
        self._initialize_model()
    
    def _initialize_model(self):
        # LLM config preparing
        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # ViT config preparing
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # VAE loading
        self.vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))

        # Bagel config preparing
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer Preparing
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # Image Transform Preparing
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Model Loading and Multi GPU Inference Preparing
        device_map = infer_auto_device_map(
            model,
            max_memory={i: self.max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(device_map)

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(self.model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=self.offload_folder
        )

        self.model = self.model.eval()
        print('Model loaded')

        self.inferencer = InterleaveInferencer(
            model=self.model, 
            vae_model=self.vae_model, 
            tokenizer=self.tokenizer, 
            vae_transform=vae_transform, 
            vit_transform=vit_transform, 
            new_token_ids=self.new_token_ids
        )
    
    def edit_image_no_think(
        self, 
        image_path: str, 
        prompt: str,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: list = [0.0, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel"
    ) -> Dict[str, Any]:

        image = Image.open(image_path)
        inference_hyper = {
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type
        }
        
        output_dict = self.inferencer(image=image, text=prompt, **inference_hyper)
        
        return {
            'image': output_dict['image']
        }
    
    def generate_caption(
        self, 
        prompt: str,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
    ) -> Dict[str, Any]:

        inference_hyper = {
            'max_think_token_n': max_think_token_n,
            'do_sample': do_sample
        }
        
        output_dict = self.inferencer(text=prompt, understanding_output=True, **inference_hyper)
        return output_dict['text']
    
 