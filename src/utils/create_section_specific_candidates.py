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

tqdm.pandas()

llama3_1_subjective_only = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a precise medical AI assistant for summarising doctor-patient dialogues into the subjective summary for the patient. The dialogues you are going to summaries are recorded conversations, they can include missing punctuations or repeating words.\
Make sure that the subjective summary contain headings like: chief complaint, history of present illness, \
review of system and social history and past history, if necessary. Additionally the turns are tagged with more precise medical intents. \
Utilize those tags to create the subjective section.<|eot_id|> \
<|start_header_id|>user<|end_header_id|> Please summarize the following conversation into a subjective summary: $$CONV$$<|eot_id|>  \
<|start_header_id|>assistant<|end_header_id|>Here is the subjective summary:"  

llama3_1_objective_exam_only = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a precise medical AI assistant for summarising doctor-patient dialogues into the objective examen summary for the patient. The dialogues you are going to summaries are recorded conversations, they can include missing punctuations or repeating words.\
Make sure that the objective examen summary contains headings like: physical examination, examination and vitals reviewed. Additionally the turns are tagged with more precise medical intents. \
Utilize those tags to create the objective examen summary.<|eot_id|> \
<|start_header_id|>user<|end_header_id|> Please summarize the following conversation into a objective examen summary: $$CONV$$<|eot_id|>  \
<|start_header_id|>assistant<|end_header_id|>Here is the objective examen summary:"  

llama3_1_objective_results_only = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a precise medical AI assistant for summarising doctor-patient dialogues into the objective results summary for the patient. The dialogues you are going to summaries are recorded conversations, they can include missing punctuations or repeating words.\
Make sure that the objective results summary contains headings like: lab results, radiology results, etc. Additionally the turns are tagged with more precise medical intents. \
Utilize those tags to create the objective results summary.<|eot_id|> \
<|start_header_id|>user<|end_header_id|> Please summarize the following conversation into a objective results summary: $$CONV$$<|eot_id|>  \
<|start_header_id|>assistant<|end_header_id|>Here is the objective results summary:"  


llama3_1_assessment_and_plan_only = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a precise medical AI assistant for summarising doctor-patient dialogues into an assessment and plan summary for the patient. The dialogues you are going to summaries are recorded conversations, they can include missing punctuations or repeating words.\
Make sure that the assessment and plan summary contains headings like: assessment, plan, assessment and plan, summary plan. Additionally the turns are tagged with more precise medical intents. \
Utilize those tags to create the assessment and plan summary.<|eot_id|> \
<|start_header_id|>user<|end_header_id|> Please summarize the following conversation into an assessment and plan summary: $$CONV$$<|eot_id|>  \
<|start_header_id|>assistant<|end_header_id|>Here is the assessment and plan summary:" 

llama3_1_section_prompts_dict = {"subjective": llama3_1_subjective_only, "objective_exam": llama3_1_objective_exam_only, "objective_results":llama3_1_objective_results_only, "assessment_and_plan": llama3_1_assessment_and_plan_only}
class DialogueFilter():
    def __init__(self,
                tokenizer,
                model,
                device: str = "cuda",
                train_file: str = "",
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        train = pd.read_json(train_file)
        all_intent_labels = sorted(train.intents.explode().dropna().unique().tolist())
        self.inverse_intent_label_lookup = {k: v for k, v in enumerate(all_intent_labels)}
        all_section_labels = sorted(train.sections.explode().dropna().unique().tolist())
        self.inverse_section_label_lookup = {k: v for k, v in enumerate(all_section_labels)}
        self.device = device
    def parse_probs(self, indices, task):
        outputs = {}
        for key, value in enumerate(list(indices[0])):
            if value.item() not in outputs.keys():
                outputs[value.item()] = [self.inverse_intent_label_lookup[indices[1][key].item()]] if task == "intent"\
                else [self.inverse_section_label_lookup[indices[1][key].item()]]
            else:
                outputs[value.item()].append(self.inverse_intent_label_lookup[indices[1][key].item()]) if task == "intent"\
                else outputs[value.item()].append(self.inverse_section_label_lookup[indices[1][key].item()])
        return outputs

    def collate_testdata(self, text):
        text = text
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=512,
        )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [
            torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]
        ]
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        return {
            "input_ids": input_ids.to(device=self.device),
            "attention_mask": attention_masks.to(device=self.device),

        }

    def filter_dialogue(self, text):
        text = text
        tokenized = self.collate_testdata(text)
        with torch.no_grad():
            encoded = self.model(tokenized)
        intent_indices = torch.where(encoded["intent"] > 0.5)
        section_indices = torch.where(encoded["section"] > 0.5)
        intent_outputs = self.parse_probs(intent_indices, "intent")
        section_outputs = self.parse_probs(section_indices, "section")
        chitchat_ids = []
        for key, intent in intent_outputs.items():
            if "Chitchat" in intent:
                chitchat_ids.append(key)
        for cc_id in reversed(chitchat_ids):
            text.remove(text[cc_id])
            if cc_id in intent_outputs.keys():
                del(intent_outputs[cc_id])
            if cc_id in section_outputs.keys():
                del(section_outputs[cc_id])
        filtered_intents = {}
        filtered_sections = {}
        for sections, intents, text_id in zip(section_outputs.values(), intent_outputs.values(), range(len(text))):
            filtered_intents[text_id] = intents
            filtered_sections[text_id] = sections
        return text, filtered_intents, filtered_sections
            

def process_snippet(role, snippet):
    snippet = snippet.strip().replace("\n", "")
    full_turn = role + ": " + snippet
    return full_turn
def capture_utterances(conversation):
    regex = r'\[(doctor|patient|patient_guest)\]\s*([\s\S]*?)(?=\[(doctor|patient|patient_guest)\]|$)'
    matches = re.findall(regex, conversation.replace("\n", ""), re.DOTALL | re.M)
    utterances = []
    i = 0
    while i < len(matches):

        doctor_role, doctor_snippet = matches[i][0:2]
        while(matches[i][2] == doctor_role):
            doctor_snippet = doctor_snippet + " " + matches[i+1][1]
            i += 1
        if matches[i][2] == "" and (i+1) >= len(matches) :
            full_doc = process_snippet(doctor_role, doctor_snippet)
            utterances.append(full_doc)
            i += 1
        else:
            patient_role, patient_snippet = matches[i+1][0:2]
            while(matches[i+1][2] in [patient_role, "patient_guest"]):
                if patient_role != matches[i+1][2]:
                    patient_snippet = patient_snippet +  " " + matches[i+1][2] + ": " + matches[i+2][1]
                    i += 1
                else:
                    patient_snippet = patient_snippet + " " + matches[i+1][1]
                    i += 1
            if doctor_role == ("patient" or "patient_guest"):
                full_doc = process_snippet(patient_role, doctor_snippet)
                full_pat = process_snippet(doctor_role, patient_snippet)
                utterances.append(full_doc + " " + full_pat)
                i += 2
            else:
                full_doc = process_snippet(doctor_role, doctor_snippet)
                full_pat = process_snippet(patient_role, patient_snippet)
                utterances.append(full_doc + " " + full_pat)
                i += 2
    return utterances



def create_section_specific_bm25_candidates(text, section, bm25_mapping, top_k, dialogue_filter, filter:bool):
    dialogue = " ".join(text)
    scores = bm25_mapping[section].get_scores(dialogue.split(" "))
    candidates  = candidate_df_dict[section].loc[np.argpartition(scores, -top_k)[-top_k:]]
    in_context_candidates = ""
    split_tkn = "<|eot_id|>"
    user_prompt = llama3_1_section_prompts_dict[section].split(split_tkn)[1].strip()
    assistant_prompt = llama3_1_section_prompts_dict[section].split(split_tkn)[2].strip()
    
    for candidate_key, candidate_text in enumerate(candidates.text.to_list()):
        if filter:
            filtered_text, filtered_intents, filtered_sections = dialogue_filter.filter_dialogue(candidate_text)
            out_string = ""
            if section == "subjective":
                for key, value in filtered_sections.items():
                    if "Subjective" in value:
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "

            if section == "objective_exam":
                for key, value in filtered_sections.items():
                    if ("Objective" in value and "Physical Examination" in filtered_intents[key]):
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "

            if section == "objective_results":
                for key, value in filtered_sections.items():
                    if "Objective" in value and ("Lab Examination" in filtered_intents[key] or "Radiology Examination" in filtered_intents[key]):
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "

            if section == "assessment_and_plan":
                for key, value in filtered_sections.items():
                    if "Assessment" in value or "Plan" in value:
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "

            tmp_user_prompt = user_prompt.replace("$$CONV$$", out_string) + split_tkn
            tmp_user_prompt += assistant_prompt + candidates.iloc[candidate_key].tgt + split_tkn
            in_context_candidates += tmp_user_prompt
        else:
            tmp_user_prompt = user_prompt.replace("$$CONV$$", " ".join(candidate_text)) + split_tkn
            tmp_user_prompt += assistant_prompt + candidates.iloc[candidate_key].tgt + split_tkn
            in_context_candidates += tmp_user_prompt

    return in_context_candidates

def create_section_specific_prompts(data, data_dict, dialogue_filter=None, filter=False, in_context=False):
    text = data["text"]
    section_prompts = {}
    for section in data_dict['clinicalnlp_taskB_test1'].keys():
        if filter:
            filtered_text, filtered_intents, filtered_sections = dialogue_filter.filter_dialogue(text)
        out_string = ""
        prompt = ""
        if section == "subjective":
            if filter:
                for key, value in filtered_sections.items():
                    if "Subjective" in value:
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "
                prompt = llama3_1_subjective_only.replace("$$CONV$$", out_string) if out_string != "" else ""
            else:
                prompt =llama3_1_subjective_only.replace("$$CONV$$", " ".join(text))
        if section == "objective_exam":
            if filter:
                for key, value in filtered_sections.items():
                    if ("Objective" in value and "Physical Examination" in filtered_intents[key]):
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "
                prompt = llama3_1_objective_exam_only.replace("$$CONV$$", out_string) if out_string != "" else ""
            else:
                prompt = llama3_1_objective_exam_only.replace("$$CONV$$", " ".join(text))
        if section == "objective_results":
            if filter:
                for key, value in filtered_sections.items():
                    if "Objective" in value and ("Lab Examination" in filtered_intents[key] or "Radiology Examination" in filtered_intents[key]):
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "
                prompt = llama3_1_objective_results_only.replace("$$CONV$$", out_string) if out_string != "" else ""
            else:
                prompt = llama3_1_objective_results_only.replace("$$CONV$$", " ".join(text))

        if section == "assessment_and_plan":
            if filter:
                for key, value in filtered_sections.items():
                    if "Assessment" in value or "Plan" in value:
                        out_string += filtered_text[key] + " (Medical Intents: " + ", ".join(filtered_intents[key]) + ") "
                prompt = llama3_1_assessment_and_plan_only.replace("$$CONV$$", out_string) if out_string != "" else ""
            else:
                prompt = llama3_1_assessment_and_plan_only.replace("$$CONV$$", " ".join(text))
        if in_context:
            split_tkn = "<|eot_id|>"
            in_context_candidates_str = create_section_specific_bm25_candidates(text, section, bm25_mapping, 3, filter)
            system_prompt, user_prompt, assistant_prompt = prompt.split(split_tkn)[0], prompt.split(split_tkn)[1], prompt.split(split_tkn)[2]
            prompt = system_prompt + split_tkn + in_context_candidates_str + user_prompt + split_tkn + assistant_prompt if prompt != "" else ""
        section_prompts[section] = prompt

    return section_prompts

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grouped_dialogues_path", type=str, default="")
    parser.add_argument("--aci_root_data_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default ="UFNLP/gatortronS")
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()
    print("Start")
    print("Load data")
    grouped_dialogues = pd.read_json(args.grouped_dialogues_path)
    test_dialogues = grouped_dialogues.loc[grouped_dialogues["split"] == "test1"].reset_index(drop=True)

    DIVISIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]
    data_dict = {"train": {}, "valid": {}, "clinicalnlp_taskB_test1": {}}
    for split in data_dict.keys():
        for div in DIVISIONS:
            data = pd.read_json(f"{args.aci_root_data_path}/{split}_{div}.json")
            data_dict[split][div] = data

    splits = ["train", "valid"]
    candidate_df_dict = {'subjective':[], 'objective_exam':[], 'objective_results':[], 'assessment_and_plan':[]}
    for split in splits:
        for section in data_dict[split].keys():
            candidate_df_dict[section].append(data_dict[split][section]["data"].apply(pd.Series))
    print("Initialize BM25")
    bm25_mapping = {}
    for section in candidate_df_dict.keys():
        candidate_df_dict[section] = pd.concat((candidate_df_dict[section]))[["src", "tgt"]].reset_index(drop=True)
        candidate_df_dict[section]["text"] = candidate_df_dict[section].src.map(capture_utterances)
        tokenized_corpus = [doc.split(" ") for doc in candidate_df_dict[section].src.to_list()]
        bm25_mapping[section] = BM25Okapi(tokenized_corpus)

    print("Load checkpoint")
    ckpt = torch.load(args.ckpt_path, weights_only=True)
    print("Load Model")
    model = MultiTaskBertClassificationModel(encoder_model_name=args.model_name, tasks={'intent':20, 'section': 5}, device=args.device, hidden_dim=args.hidden_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.encoder.to(args.device)
    print("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Initialize Filter")
    dialogue_filter = DialogueFilter(tokenizer, model, args.device, args.train_file)
    print("Start creating")
    test_dialogues["in_context_section_prompts"] = test_dialogues.data.progress_map(lambda x: create_section_specific_prompts(x, data_dict, dialogue_filter, True, True))
    print("Done")
    print(f"Saving to {args.output_file}")
    test_dialogues["in_context_section_prompts"].to_json(args.output_file, index=False)
