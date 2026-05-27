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

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import os
from PIL import Image


def generate_image_paths(img_dir, output_csv, batch_size=1, device=None):
    img_file_list = os.listdir(img_dir)
    image_id_list, image_path_list = [], []

    for i in tqdm(range(0, len(img_file_list), batch_size)):
        batch_files = img_file_list[i:i + batch_size]

        for img_file in batch_files:
            img_path = os.path.join(img_dir, img_file)
            image_id_list.append(os.path.splitext(img_file)[0])
            image_path_list.append(img_path)

    df = pd.DataFrame({"image_id": image_id_list, "image_path": image_path_list})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Finished, save in: {output_csv}")
    print(f"Total images processed: {len(image_id_list)}")


def main():
    parser = argparse.ArgumentParser(description="Image Path Collector")
    parser.add_argument("--dataset_name", type=str, default="CIRCO", choices=["FASHIONIQ", "CIRCO", "CIRR"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (not used)")
    args = parser.parse_args()

    DATASET_IMAGE_PATH = {
        "FASHIONIQ": "/home/user/datasets/FASHIONIQ/images",
        "CIRCO": "/home/user/datasets/CIRCO/COCO2017_unlabeled/unlabeled2017",
        "CIRR": "/home/user/datasets/CIRR/test1"
    }

    args.output_csv = f"{args.dataset_name}_image_paths.csv"

    output_csv = os.path.join("/home/user/datasets", args.dataset_name, "preload/image_paths", args.output_csv)

    generate_image_paths(DATASET_IMAGE_PATH[args.dataset_name], output_csv, args.batch_size, args.device)

if __name__ == "__main__":
    main()