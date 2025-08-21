from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
import re
from ast import literal_eval
from multi_task_bert import MultiTaskBertClassificationModel
from rank_bm25 import BM25Okapi
from argparse import ArgumentParser
from utils.promptutils import capture_utterances, DialogueFilter, generate_section_summaries, PromptBuilder
from utils.promptenum import LLamaEnum, PhiEnum, QwenEnum
import os
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inference_server", type=str, default="http://10.244.4.177:80")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--encoder_model_name", type=str)
    parser.add_argument("--train_file", type=str, default ="/Users/toroe/Workspace/medical_intent_verification/dataset/train_doc_only_synthetic_stratified_all.json")
    parser.add_argument("--intent_train_file", type=str, default ="/Users/toroe/Workspace/medical_intent_verification/dataset/train_doc_only_synthetic_stratified_all.json")
    parser.add_argument("--dev_file", type=str,default ="/Users/toroe/Workspace/medical_intent_verification/dataset/dev_doc_only_synthetic_stratified_all.json")
    parser.add_argument("--test_file", type=str, default ="/Users/toroe/Workspace/medical_intent_verification/dataset/test_doc_only_synthetic_stratified_all.json")
    parser.add_argument("--grouped_dialogues_path", type=str, default="/Users/toroe/Workspace/medical_intent_verification/data/all_dialogues_annotated.json")
    parser.add_argument("--aci_root_data_path", type=str, default="/Users/toroe/Workspace/aci-bench/data/challenge_data_json")
    parser.add_argument("--summarise_section", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--in_context", action="store_true")
    parser.add_argument("--full_note", action="store_true")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--results_path", type=str, default="/pvc/experiments")
    args = parser.parse_args()
    print("Start up")
    hf_token = "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"
    hf_cache = "/pvc/huggingface_cache/models"
    client = InferenceClient(model=args.inference_server, token=hf_token)
    open_ai_token = "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"
    model_name = args.model_id.split("/")[-1]
    
    out_path = f"{args.results_path}/{model_name}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Load data")
    cleaned_data_train = pd.read_csv(args.train_file)
    cleaned_data_val = pd.read_csv(args.dev_file)
    cleaned_data_test_1 = pd.read_csv(args.test_file)
    grouped_dialogues = pd.read_json(args.grouped_dialogues_path)
    test_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "test1"].reset_index(drop=True)
    train_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "train"].reset_index(drop=True)
    val_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "val"].reset_index(drop=True)
    experiment_output = {}
    bm25_mapping = {}
    data_dict = {}
    candidate_df_dict = {}
    print(f"Filter: {args.filter}")
    print(f"In-context: {args.in_context}")
    print(f"Full-note: {args.full_note}")
    print(f"Sections: {args.summarise_section}")
    if args.filter:
        print("Load filter")
        ckpt = torch.load(args.ckpt_path, mmap=True, weights_only=True, map_location=args.device)
        model = MultiTaskBertClassificationModel(encoder_model_name=args.encoder_model_name, tasks={'intent':20}, device=args.device, hidden_dim=1024)
        model.load_state_dict(ckpt["state_dict"])
        model.encoder.to(args.device)
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        dialogue_filter = DialogueFilter(encoder_tokenizer, model, args.device, args.intent_train_file)
    else:
        dialogue_filter = None
    if args.full_note and args.in_context:
        candidate_data = pd.concat((cleaned_data_train, cleaned_data_val)).reset_index(drop=True)
        tokenized_corpus = [doc.split(" ") for doc in candidate_data.dialogue.to_list()]
        bm25 = BM25Okapi(tokenized_corpus)

    if args.summarise_section:
        DIVISIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]
        data_dict = {"train": {}, "valid": {}, "clinicalnlp_taskB_test1": {}}
        for split in data_dict.keys():
            for div in DIVISIONS:
                data = pd.read_json(f"{args.aci_root_data_path}/{split}_{div}.json")
                data_dict[split][div] = data
        if args.in_context:
            splits = ["train", "valid"]
            candidate_df_dict = {'subjective':[], 'objective_exam':[], 'objective_results':[], 'assessment_and_plan':[]}
            for split in splits:
                for section in data_dict[split].keys():
                    candidate_df_dict[section].append(data_dict[split][section]["data"].apply(pd.Series))
            print("Create section BM25")
            
            for section in candidate_df_dict.keys():
                candidate_df_dict[section] = pd.concat((candidate_df_dict[section]))[["src", "tgt"]].reset_index(drop=True)
                candidate_df_dict[section]["text"] = candidate_df_dict[section].src.map(capture_utterances)
                tokenized_corpus = [doc.split(" ") for doc in candidate_df_dict[section].src.to_list()]
                bm25_mapping[section] = BM25Okapi(tokenized_corpus)
    builder = PromptBuilder(model_name.lower(), dialogue_filter, bm25_mapping, 3, bm25, candidate_data, data_dict, candidate_df_dict, grouped_dialogues, args.device, args.in_context, open_ai_token)
    if "gpt" not in args.model_id:
        if not os.path.isfile(f"{out_path}/unfiltered_full_note_no_context.json"):
            print("Start Full Note no-context")
            experiment_output["unfiltered_full_note_no_context"] = cleaned_data_test_1.dialogue.map(
                    lambda x: client.text_generation(builder.prompt.replace("$$CONV$$", x), max_new_tokens=512, temperature=0.2))
            experiment_output["unfiltered_full_note_no_context"].to_json(f"{out_path}/unfiltered_full_note_no_context.json", index=False)
        else:
            print(f"{out_path}/unfiltered_full_note_no_context.json already present")
        if not os.path.isfile(f"{out_path}/unfiltered_full_note_in_context.json"):
            print("Start Full Note in-context")
            experiment_output["unfiltered_full_note_in_context"] = cleaned_data_test_1.dialogue.map(
                lambda x: client.text_generation(builder.prepare_in_context_summary_prompt(x), max_new_tokens=512, temperature=0.2))
            experiment_output["unfiltered_full_note_in_context"].to_json(f"{out_path}/unfiltered_full_note_in_context.json", index=False)
        else:
            print(f"{out_path}/unfiltered_full_note_in_context.json already present")

        if not os.path.isfile(f"{out_path}/filtered_full_note_in_context.json"):
            print("Start Full Note Filtered in-context")
            filtered_full_note_prompts_in_context = test_dialogues.data.map(
                lambda x: builder.prepare_filtered_prompt(x, in_context=True))
            experiment_output["filtered_full_note_in_context"] = filtered_full_note_prompts_in_context.map(
                lambda x: client.text_generation(x, max_new_tokens=512, temperature=0.2))
            experiment_output["filtered_full_note_in_context"].to_json(f"{out_path}/filtered_full_note_in_context.json", index=False)
        else:
            print(f"{out_path}/filtered_full_note_in_context.json already present")
            
        if not os.path.isfile(f"{out_path}/filtered_full_note_no_context.json"):
            print("Start Full Note Filtered in-context")
            filtered_full_note_prompts_in_context = test_dialogues.data.map(
                lambda x: builder.prepare_filtered_prompt(x, in_context=False))
            experiment_output["filtered_full_note_no_context"] = filtered_full_note_prompts_in_context.map(
                lambda x: client.text_generation(x, max_new_tokens=512, temperature=0.2))
            experiment_output["filtered_full_note_no_context"].to_json(f"{out_path}/filtered_full_note_no_context.json", index=False)
        else:
            print(f"{out_path}/filtered_full_note_no_context.json already present")
        print("Done")
            

        if not os.path.isfile(f"{out_path}/unfiltered_section_in_context.json"):
            print("Start sections in-context unfiltered")
            unfiltered_section_prompts_in_context = test_dialogues.data.map(
                lambda x: builder.create_section_specific_prompts(x, filter=False, in_context=True))
            experiment_output["unfiltered_section_in_context"] = unfiltered_section_prompts_in_context.map(lambda x: generate_section_summaries(x, client))
            experiment_output["unfiltered_section_in_context"].to_json(f"{out_path}/unfiltered_section_in_context.json", index=False)
        else:
            print(f"{out_path}/unfiltered_section_in_context.json already present")

        if not os.path.isfile(f"{out_path}/filtered_section_in_context.json"):
            print("Start sections in-context filtered")
            filtered_section_prompts_in_context = test_dialogues.data.map(
                lambda x: builder.create_section_specific_prompts(x, filter=True, in_context=True))
            experiment_output["filtered_section_in_context"] = filtered_section_prompts_in_context.map(lambda x: generate_section_summaries(x, client))
            experiment_output["filtered_section_in_context"].to_json(f"{out_path}/filtered_section_in_context.json", index=False)
        else:
            print(f"{out_path}/filtered_section_in_context.json alrey present")

        if not os.path.isfile(f"{out_path}/unfiltered_section_no_context.json"):
            print("Start sections no-context unfiltered")
            unfiltered_section_prompts_no_context = test_dialogues.data.map(lambda x: builder.create_section_specific_prompts(x, filter=False, in_context=False))
            experiment_output["unfiltered_section_no_context"] = unfiltered_section_prompts_no_context.map(lambda x: generate_section_summaries(x, client))
            experiment_output["unfiltered_section_no_context"].to_json(f"{out_path}/unfiltered_section_no_context.json", index=False)
        else:
            print(f"{out_path}/unfiltered_section_no_context.json already present")

        if not os.path.isfile(f"{out_path}/filtered_section_no_context.json"):
            print("Start sections no-context filtered")
            filtered_section_prompts_no_context = test_dialogues.data.map(lambda x: builder.create_section_specific_prompts(x, filter=True, in_context=False))
            experiment_output["filtered_section_no_context"] = filtered_section_prompts_no_context.map(lambda x: generate_section_summaries(x, client))
            experiment_output["filtered_section_no_context"].to_json(f"{out_path}/filtered_section_no_context.json", index=False)
        else:
            print(f"{out_path}/filtered_section_no_context.json already present")
    else:
        no_context_full_note_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_prompt(x, False, False))
        gpt_full_note_no_context_out = no_context_full_note_prompts.map(lambda x: builder.infer_gpt(x, False))
        gpt_full_note_no_context_out.to_json(f"{args.results_path}/unfiltered_full_note_no_context.json", index=False)
        
        """
        filtered_in_context_section_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_section_specific_prompts(x, True, True))
        filtered_in_context_section_output = filtered_in_context_section_prompts.map(lambda x: builder.infer_gpt(x, True))
        filtered_in_context_section_output.to_json(f"{args.results_path}/filtered_section_in_context.json", index=False)

        filtered_no_context_full_note_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_filtered_prompt(x,False))
        filtered_no_context_full_note_output = filtered_no_context_full_note_prompts.map(lambda x: builder.infer_gpt(x, False))
        filtered_no_context_full_note_output.to_json(f"{args.results_path}/filtered_full_note_no_context.json", index=False)

        filtered_in_context_full_note_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_filtered_prompt(x,True))
        filtered_in_context_full_note_output = filtered_in_context_full_note_prompts.map(lambda x: builder.infer_gpt(x, False))
        filtered_in_context_full_note_output.to_json(f"{args.results_path}/filtered_full_note_in_context.json", index=False)

        in_context_section_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_section_specific_prompts(x, False, True))
        in_context_section_output = in_context_section_prompts.map(lambda x: builder.infer_gpt(x, True))
        in_context_section_output.to_json(f"{args.results_path}/unfiltered_section_in_context.json", index=False)

        no_context_section_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_section_specific_prompts(x, False, False))
        no_context_section_output = no_context_section_prompts.map(lambda x: builder.infer_gpt(x, True))
        no_context_section_output.to_json(f"{args.results_path}/unfiltered_section_no_context.json", index=False)

        in_context_full_note_prompts = cleaned_data_test_1.dialogue.map(lambda x: builder.prepare_gpt_prompt(x, False, True))
        gpt_full_note_in_context_out = in_context_full_note_prompts.map(builder.infer_gpt)
        gpt_full_note_in_context_out.to_json(f"{args.results_path}/unfiltered_full_note_in_context.json", index=False)

        """
    print("Done all.")

