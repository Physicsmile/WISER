import argparse
import prompts

def parse_arguments():
    parser = argparse.ArgumentParser()

    # === Base Arguments ===
    parser.add_argument("--exp_name", 
                        type=str, 
                        help="Experiment to evaluate")
    parser.add_argument("--device", 
                        type=int, 
                        default=0, 
                        help="GPU ID to use.")
    parser.add_argument("--preload", 
                        nargs='+', 
                        type=str, 
                        default=['captions', 'mods'], 
                        help="List of properties to preload (computed once before).")

    # === Base Model Choices ===
    parser.add_argument("--clip", 
                        type=str, 
                        default='ViT-B-32', 
                        choices=['ViT-Base/32', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 
                                 'ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use.")
    # === Dataset Arguments ===
    parser.add_argument("--dataset", 
                        default="fashioniq_dress", 
                        type=str, 
                        required=False, 
                        choices=['cirr', 'circo', 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 
                                 'genecis_focus_object'],
                        help="Dataset to use")
    parser.add_argument("--split", 
                        type=str, 
                        default='val', 
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols like cirr/circo.')
    parser.add_argument("--dataset_path", 
                        default="../datasets/FASHIONIQ", 
                        type=str, 
                        required=False,
                        help="Path to the dataset")
    parser.add_argument("--preprocess-type", 
                        default="targetpad", 
                        type=str, 
                        choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")

    # === LLM & BLIP Prompt Arguments ===
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--llm_prompt", 
                        default='prompts.structural_modifier_prompt_fashion', 
                        type=str, 
                        choices=available_prompts,
                        help='Base prompt to use to probe the LLM. Must be available in prompts.py') 
    parser.add_argument("--openai_key", 
                        default="<your_openai_key_here>", 
                        type=str,
                        help='Account key for OpenAI LLM usage.')

    # === Caption Checking Arguments ===
    parser.add_argument("--max_check_num", 
                        default=1, 
                        type=int,
                        help='Maximum number of times the modified captions need to be checked.')

    # === LLM Model Arguments ===
    parser.add_argument("--LLM_model_name", 
                        default="gpt-4o-mini", 
                        type=str, choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "qwen-turbo"],
                        help='LLM model name to use.')
    parser.add_argument("--Check_LLM_model_name", 
                        default="gpt_4o_mini", 
                        type=str, 
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "qwen-turbo"],
                        help='LLM model name to check modified captions.')

    # === Text-to-Image Retrieval Arguments ===
    parser.add_argument("--retrieval", 
                        type=str, 
                        default='default', 
                        choices=['default'],
                        help='Type of T2I Retrieval method.')

    return parser.parse_args()
