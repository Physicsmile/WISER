import os
import argparse
import warnings
import logging
import torch
import pandas as pd
from tqdm import tqdm
import os
from typing import List, Dict

import argparse
import lavis
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import os
from PIL import Image


def load_model_and_processor(captioner, device):

    if captioner == "blip2_t5":
        model, processor, _ = lavis.models.load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
        model = model.float()
        model.maybe_autocast = lambda dtype=None: torch.no_grad()
        tokenizer = None
    else:
        raise ValueError(f"Unsupported model type: {captioner}")

    return processor, model, tokenizer


def generate_caption(blip_transform, model_type, model, processor, tokenizer, img_path, device):
    prompt = "Describe the image in complete detail. You must especially focus on all the objects in the image."

    if model_type == "blip2_t5":
        img = blip_transform(Image.open(img_path).convert('RGB'))
        img = img.unsqueeze(0).to(device)
        caption = model.generate({'image': img, "prompt": prompt})
        print("caption",caption[0])

        return caption[0]

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_captions(img_dir, output_csv, captioner, batch_size=1, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model, tokenizer = load_model_and_processor(captioner, device)
    img_file_list = os.listdir(img_dir)
    image_id_list, generated_text_list = [], []

    for i in tqdm(range(0, len(img_file_list), batch_size)):
        batch_files = img_file_list[i:i + batch_size]

        for img_file in batch_files:
            img_path = os.path.join(img_dir, img_file)
            try:
                caption = generate_caption(processor["eval"], captioner, model, processor, tokenizer, img_path, device)
            except Exception as e:
                caption = f"Error: {e}"

            image_id_list.append(os.path.splitext(img_file)[0])
            generated_text_list.append(caption)

        torch.cuda.empty_cache()

    df = pd.DataFrame({"image_id": image_id_list, "generated_text": generated_text_list})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Finished, save in: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument("--dataset_name", type=str, default="CIRCO", choices=["FASHIONIQ", "CIRCO", "CIRR"], help="Dataset name")
    parser.add_argument("--captioner", type=str, default='blip2_t5', choices=["blip2_t5"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()

    DATASET_IMAGE_PATH = {
        "FASHIONIQ": "/home/user/datasets/FASHIONIQ/images",
        "CIRCO": "/home/user/datasets/CIRCO/COCO2017_unlabeled/unlabeled2017",
        "CIRR": "/home/user/datasets/CIRR/test1"
    }

    args.output_csv = f"{args.dataset_name}_{args.captioner}_captions.csv"

    output_csv = os.path.join("/home/user/datasets", args.dataset_name, "preload/image_captions", args.output_csv)

    generate_captions(DATASET_IMAGE_PATH[args.dataset_name], output_csv, args.captioner, args.batch_size, args.device)

if __name__ == "__main__":
    main()
