import torch
import pandas as pd
from rank_bm25 import BM25Okapi
from enum import Enum
from argparse import ArgumentParser
import re
from transformers import AutoModel, AutoTokenizer
from multi_task_bert import MultiTaskBertClassificationModel
from src.encoder_models.datamodule import MultiTaskDataModule
import numpy as np
from tqdm import tqdm
import os
from src.utils.create_section_specific_candidates import capture_utterances, DialogueFilter, create_section_specific_prompts
def create_filtered_fine_tune_data(data,  dialog_filter, data_dict=None):
    text = data["text"]
    filtered_text, filtered_intents, filtered_sections = dialog_filter.filter_dialogue(text)
    filtered_dialogue = {}
    if data_dict:
        for section in data_dict['clinicalnlp_taskB_test1'].keys():
            out_string = ""
            if section == "subjective":
                    for key, value in filtered_sections.items():
                        if "Subjective" in value:
                            out_string += filtered_text[key] + " "
            if section == "objective_exam":
                for key, value in filtered_sections.items():
                    if "Objective" in value and ("Physical Examination" in filtered_intents[key]) or ("Lab Examination" not in filtered_intents[key] and "Radiology Examination" not in filtered_intents[key]):
                        out_string += filtered_text[key] + " "
            if section == "objective_results":
                for key, value in filtered_sections.items():
                    if "Objective" in value and ("Lab Examination" in filtered_intents[key] or "Radiology Examination" in filtered_intents[key]):
                        out_string += filtered_text[key] + " "
            if section == "assessment_and_plan":
                for key, value in filtered_sections.items():
                    if "Assessment" in value or "Plan" in value:
                        out_string += filtered_text[key] + " "     
            filtered_dialogue[section] = out_string 
        return filtered_dialogue
    else:
        return filtered_text
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grouped_dialogues_path", type=str, default="")
    parser.add_argument("--aci_root_data_path", type=str, default="")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--dev_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--model_name", type=str, default ="UFNLP/gatortronS")
    parser.add_argument("--encoder_train_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()
    print("Start")
    print("Load data")
    grouped_dialogues = pd.read_json(args.grouped_dialogues_path)
    train_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "train"].reset_index(drop=True)
    val_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "val"].reset_index(drop=True)
    test_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "test1"].reset_index(drop=True)

    train_data = pd.read_csv(args.train_file)
    valid_data = pd.read_csv(args.dev_file)
    test_data = pd.read_csv(args.test_file)
    DIVISIONS = ["subjective", "objective_exam", "assessment_and_plan", "objective_results"]#
    data_dict = {"train": {}, "valid": {}, "clinicalnlp_taskB_test1": {}}
    for split in data_dict.keys():
        for div in DIVISIONS:
            data = pd.read_json(f"{args.aci_root_data_path}/{split}_{div}.json")
            data_dict[split][div] = data

  

    print("Load checkpoint")
    ckpt = torch.load(args.ckpt_path, weights_only=True)
    print("Load Model")
    model = MultiTaskBertClassificationModel(encoder_model_name=args.model_name, tasks={'intent':20, 'section': 5}, device=args.device, hidden_dim=args.hidden_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.encoder.to(args.device)
    print("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Initialize Filter")
    dialogue_filter = DialogueFilter(tokenizer, model, args.device, args.encoder_train_file)
    for split in data_dict.keys():
        tmp = {}
        if split == "train":  
            filtered_dialogues = train_dialogues.data.apply(lambda x: create_filtered_fine_tune_data(x, dialogue_filter))
            filtered_dialogues = filtered_dialogues.map(lambda x: " ".join(x))
            notes = train_data.loc[filtered_dialogues.index].note

        elif split == "valid":
            filtered_dialogues = val_dialogues.data.apply(lambda x: create_filtered_fine_tune_data(x, dialogue_filter))
            filtered_dialogues = filtered_dialogues.map(lambda x: " ".join(x))
            notes = valid_data.loc[filtered_dialogues.index].note
        elif 'clinicalnlp_taskB_test1':
            filtered_dialogues = test_dialogues.data.apply(lambda x: create_filtered_fine_tune_data(x, dialogue_filter))
            filtered_dialogues = filtered_dialogues.map(lambda x: " ".join(x))
            notes = test_data.loc[filtered_dialogues.index].note
        pd.DataFrame.from_dict({"dialogue":filtered_dialogues, "note":notes}).to_csv(args.output_path + f"/{split}.csv", index=False)
