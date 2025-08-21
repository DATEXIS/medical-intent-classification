import numpy as np
import pandas as pd
import torch
import re
from utils.promptenum import LLamaEnum, PhiEnum, QwenEnum, GPTEnum
from openai import OpenAI


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
        self.section_mapping = {
            "Acute Symptoms": "Subjective",
            "Personal History": "Subjective",
            "Therapeutic History": "Subjective",
            "Drug History": "Subjective",
            "Vegetative History": "Subjective",
            "Greetings":"Subjective",
            "Family History": "Subjective",
            "Other Socials": "Subjective",
            'Physical Examination': "Objective",
            'Lab Examination': "Objective",
            'Lab Examination': "Objective",
            'Acute Assessment': "Assessment",
            'Reassessment': "Assessment",
            'Discussion': "Plan",
            'Diagnostic Testing': "Plan",
            'Other Treatments': "Plan",
            'Follow-up': "Plan",
            'Referral': "Plan",
            'Medication': "Plan",
            'Chitchat': "Null",
        }
        

 
 
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
        if type(text) == str:
            text = capture_utterances(text)
        tokenized = self.collate_testdata(text)
        with torch.no_grad():
            encoded = self.model(tokenized)

        intent_indices = torch.where(encoded["intent"] > 0.5)

        intent_outputs = self.parse_probs(intent_indices, "intent")

        chitchat_ids = []
        for key, intent in intent_outputs.items():
            if "Chitchat" in intent:
                chitchat_ids.append(key)
        for cc_id in reversed(chitchat_ids):
            text.remove(text[cc_id])
            if cc_id in intent_outputs.keys():
                del(intent_outputs[cc_id])
        filtered_intents = {}
        filtered_sections = {}
        for intents, text_id in zip(intent_outputs.values(), range(len(text))):
            sections = [self.section_mapping[intent] for intent in intents]
            filtered_intents[text_id] = intents
            filtered_sections[text_id] = sections
        return text, filtered_intents, filtered_sections
            
class PromptBuilder():
    def __init__(self,
                model_name,
                dialog_filter,
                bm25_mapping,
                top_k,
                full_note_bm25,
                candidate_df,
                section_data_dict,
                section_candidate_df_dict,
                grouped_dialogues,
                device: str = "cuda",
                in_context: bool = False,
                open_ai_token: str = ""
                ):
        super().__init__()
        print("Load filter")
        self.section_data_dict = section_data_dict
        self.section_candidate_df_dict = section_candidate_df_dict
        self.candidate_data = candidate_df
        self.grouped_dialogues = grouped_dialogues
        self.dialogue_filter = dialog_filter
        self.full_note_bm25 = full_note_bm25
        self.bm25_mapping = bm25_mapping
        self.top_k = top_k
        self.in_context = in_context
        prompt_token_dict = {}
        if "gpt" not in model_name:
            if "llama" in model_name.lower():
                self.prompt_enum = LLamaEnum
                self.prompt = LLamaEnum.instruct_prompt.value
                prompt_token_dict["user"]="<|start_header_id|>user<|end_header_id|>"
                prompt_token_dict["assistant"] = "<|start_header_id|>assistant<|end_header_id|>"
                prompt_token_dict["eot"]  = "<|eot_id|>"
            elif "qwen" in model_name.lower():
                self.prompt_enum = QwenEnum
                self.prompt = QwenEnum.instruct_prompt.value
                prompt_token_dict["user"] ="<|im_start|>user"
                prompt_token_dict["assistant"] = "<|im_start|>assistant"
                prompt_token_dict["eot"] = "<|im_end|>"
            elif "phi" in model_name.lower():
                self.prompt_enum = PhiEnum
                self.prompt = PhiEnum.instruct_prompt.value
                prompt_token_dict["user"] ="<|im_start|>user<|im_sep|>"
                prompt_token_dict["assistant"] = "<|im_start|>assistant<|im_sep|>"
                prompt_token_dict["eot"] = "<|im_end|>"
            self.prompt_token_dict = prompt_token_dict   
            self.section_prompts_dict = {"subjective": self.prompt_enum.subjective_only.value,
                                "objective_exam": self.prompt_enum.objective_exam_only.value,
                                "objective_results": self.prompt_enum.objective_results_only.value,
                                "assessment_and_plan": self.prompt_enum.assessment_and_plan_only.value}
        else:
            self.prompt_enum = GPTEnum
            self.client = OpenAI(
    api_key=open_ai_token
    )
            self.section_prompts_dict = {
                                "subjective": {"dev": self.prompt_enum.subjective_developer.value ,"user":self.prompt_enum.subjective_user.value ,"assistant": self.prompt_enum.subjective_assistant.value},
                                "objective_exam": {"dev": self.prompt_enum.objective_exam_developer.value ,"user":self.prompt_enum.objective_exam_user.value ,"assistant": self.prompt_enum.objective_exam_assistant.value},
                                "objective_results": {"dev": self.prompt_enum.objective_results_developer.value ,"user":self.prompt_enum.objective_results_user.value ,"assistant": self.prompt_enum.objective_results_assistant.value},
                                "assessment_and_plan": {"dev": self.prompt_enum.assessment_and_plan_developer.value ,"user":self.prompt_enum.assessment_and_plan_user.value ,"assistant": self.prompt_enum.assessment_and_plan_assistant.value}
                                }
        
    def create_section_specific_bm25_candidates(self, text, section, filter):
        dialogue = " ".join(text) if type(text) == list else text
        scores = self.bm25_mapping[section].get_scores(dialogue.split(" "))
        candidates  = self.section_candidate_df_dict[section].loc[np.argpartition(scores, -self.top_k)[-self.top_k:]]
        in_context_candidates = ""
        gpt_prompts = []
        for candidate_key, candidate_text in enumerate(candidates.text.to_list()):
            if filter:
                filtered_text, filtered_intents, filtered_sections = self.dialogue_filter.filter_dialogue(candidate_text)
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
                if self.prompt_enum != GPTEnum:
                    in_context_candidates += self.prompt_token_dict["user"] + " Please summarize the following conversation into a " + section.replace("_", " ") + " summary: " + out_string + self.prompt_token_dict["eot"] + self.prompt_token_dict["assistant"] + " Here is the " +  section.replace("_", " ") + " summary: " + candidates.iloc[candidate_key].tgt + self.prompt_token_dict["eot"]
                else:
                    gpt_prompts.append({
                "role": "user",
                "content":self.section_prompts_dict[section]["user"].replace("$$CONV$$", out_string)
                })
                    gpt_prompts.append({
                    "role": "assistant",
                    "content": self.section_prompts_dict[section]["assistant"] + " " + candidates.iloc[candidate_key].tgt
                    })
            else:
                if self.prompt_enum != GPTEnum:
                    in_context_candidates += self.prompt_token_dict["user"] + " Please summarize the following conversation into a " + section.replace("_", " ") + " summary: " + " ".join(candidate_text)+ self.prompt_token_dict["eot"] + self.prompt_token_dict["assistant"] + " Here is the " +  section.replace("_", " ") + " summary: " + candidates.iloc[candidate_key].tgt + self.prompt_token_dict["eot"]
                else:
                    gpt_prompts.append({
                "role": "user",
                "content": self.section_prompts_dict[section]["user"].replace("$$CONV$$"," ".join(candidate_text))
                })
                    gpt_prompts.append({
                "role": "assistant",
                "content":self.section_prompts_dict[section]["assistant"] + " " + candidates.iloc[candidate_key].tgt
                })
                
    
        return in_context_candidates if self.prompt_enum != GPTEnum else gpt_prompts
    #GPT



    def create_section_specific_prompts(self, data, filter, in_context, oracle):
        text = data["text"]
        section_prompts = {}
        for section in self.section_data_dict['clinicalnlp_taskB_test1'].keys():
            out_string = ""
            prompt = ""
            if filter:
                if oracle:
                    filtered_text, filtered_intents, filtered_sections = data["text"], data["intents"], data["sections"]
                    for intents in data["intents"]:
                        for intent in intents:
                            if intent == "Start-Topic" or intent == "Inside-Topic":
                                intents.remove(intent)
                    filtered_sections = {key:value for key, value in enumerate(filtered_sections)}
                else:
                    filtered_text, filtered_intents, filtered_sections = self.dialogue_filter.filter_dialogue(text)
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
                prompt = self.section_prompts_dict[section].replace("$$CONV$$", out_string) if out_string != "" else ""
            else:
                prompt = self.section_prompts_dict[section].replace("$$CONV$$", " ".join(text))
            if in_context:
                if prompt != "":
                    in_context_candidates_str = self.create_section_specific_bm25_candidates(text, section, filter)
                system_prompt, tail_prompt = prompt.split(self.prompt_token_dict["user"])[0], prompt.split(self.prompt_token_dict["user"])[-1]
                prompt = system_prompt + in_context_candidates_str + self.prompt_token_dict["user"] + tail_prompt
                
            section_prompts[section] = prompt
        return section_prompts
    
    def prepare_in_context_summary_prompt(self, text):
        scores = self.full_note_bm25.get_scores(text.split(" "))
        candidates  = self.candidate_data.loc[np.argpartition(scores, -self.top_k)[-self.top_k:]]
        candidates["prompt"] = candidates.apply(
            lambda x: 
            self.prompt_token_dict["user"] + " Please summarize the following conversation into a medical note: " + x['dialogue'] + self.prompt_token_dict["eot"] + self.prompt_token_dict["assistant"] + " Here is the medical note: " + x['note'] + self.prompt_token_dict["eot"],
            axis=1
            )
        system_prompt = self.prompt_enum.instruct_prompt.value.split(self.prompt_token_dict["user"])[0]
        candidate_string = " ".join(candidates["prompt"].to_list())
        target_string = (self.prompt_token_dict["user"] + " Please summarize the following conversation into a medical note: $$CONV$$" + self.prompt_token_dict["eot"] + self.prompt_token_dict["assistant"]+ " Here is the medical note:").replace("$$CONV$$", text)
        return system_prompt + candidate_string + target_string

    def prepare_candidates(self, candidate):
        enc_id = candidate.encounter_id
        processed_candidate = self.grouped_dialogues.loc[self.grouped_dialogues.encounter_id == enc_id].data.item()
        candidate_note = self.candidate_data.loc[self.candidate_data.encounter_id == enc_id].note.item()
        if self.prompt_enum != GPTEnum:
            return_str = self.prompt_token_dict["user"] + " Please summarize the following conversation into a medical note:"
        else:
            return_str = "Please summarize the following conversation into a medical note:"
        for text, intents, sections in zip(processed_candidate["text"], processed_candidate["intents"], processed_candidate["sections"]):
            if "Start-Topic" in intents or "Inside-Topic" in intents:
                intents.remove("Start-Topic") if "Start-Topic" in intents\
                else intents.remove("Inside-Topic")
            if "Null" in sections:
                sections.remove("Null")
            intent_str = "[Intents]: " + ", ".join(intents) if intents != [] else "[Intents]: "
            section_str = "[Sections]: " + ", ".join(sections) if sections != [] else "[Sections]: "
            return_str = return_str + " " + text + " " + section_str + " " + intent_str
        if self.prompt_enum != GPTEnum:
            return_str += self.prompt_token_dict["eot"] + self.prompt_token_dict["assistant"] + " Here is the medical note: " + candidate_note + self.prompt_token_dict["eot"]
            return return_str
        else:
            gpt_prompt = {
                "role": "user",
                "content": [{ "type": "text", "text": return_str}]
                },{
                "role": "assistant",
                "content": [{ "type": "text", "text":  "Here is the medical note: " + candidate_note}]
                }
            return gpt_prompt


    def append_outputs_to_text(self, outputs, text, task):

        for key, output in outputs.items():
            if "Null" in output:
                output.remove("Null")
            to_append = " [Sections]: " + ", ".join(output) if task == "section"\
            else " [Intents]: "+ ", ".join(output)
            text[key] += to_append
        return text

    def prepare_filtered_prompt(self, data, in_context, type):
        text = data["text"]
        filtered_text, filtered_intents, filtered_sections = self.dialogue_filter.filter_dialogue(text)
        if len(filtered_intents) != len(filtered_text) or len(filtered_sections) != len(filtered_text):
            print(data["encounter_id"])
        filtered_text = self.append_outputs_to_text(filtered_sections, filtered_text, "section")
        filtered_text = self.append_outputs_to_text(filtered_intents, filtered_text, "intent")
        filtered_text = " ".join(filtered_text)
        if in_context:
            dialogue = " ".join(data["text"])
            scores = self.full_note_bm25.get_scores(dialogue.split(" "))
            candidates  = self.candidate_data.loc[np.argpartition(scores, -self.top_k)[-self.top_k:]]
            candidates["prompt_example"] = candidates.apply(lambda x: self.prepare_candidates(x), axis=1)
            candidate_string = " ".join(candidates["prompt_example"].to_list())
            filtered_text = candidate_string + " " + filtered_text
        return self.prompt_enum.full_note_section_intents.value.replace("$$CONV$$", filtered_text)
       
   
    def prepare_gpt_prompt(self, dialogue, filter, in_context):
        messages = []
        if not filter:
            messages.append({"role": "developer", "content": GPTEnum.instruct_prompt_developer.value})

            if in_context:
                scores = self.full_note_bm25.get_scores(dialogue.split(" "))
                candidates  = self.candidate_data.loc[np.argpartition(scores, -self.top_k)[-self.top_k:]]
                candidates["prompt"] = candidates.apply(lambda x: 
                                [
                                    {"role": "user","content": GPTEnum.instruct_user.value.replace("$$CONV$$", x['dialogue'])},
                                    {"role": "assistant","content": GPTEnum.instruct_assistant.value + x['note']}
                                ],
                                axis=1)
                candidate_list = [x for xs in candidates["prompt"].to_list() for x in xs]
                for candidate_input in candidate_list:
                    messages.append(candidate_input)

            messages.append({"role": "user","content": GPTEnum.instruct_user.value.replace("$$CONV$$", dialogue)})
            messages.append({"role": "assistant","content": GPTEnum.instruct_assistant.value})
        else:
            pass
        return messages
    
    def prepare_gpt_candidates(self, candidate):
        enc_id = candidate.encounter_id
        processed_candidate = self.grouped_dialogues.loc[self.grouped_dialogues.encounter_id == enc_id].data.item()
        candidate_note = self.candidate_data.loc[self.candidate_data.encounter_id == enc_id].note.item()
        return_str = ""
        for text, intents, sections in zip(processed_candidate["text"], processed_candidate["intents"], processed_candidate["sections"]):
            if "Start-Topic" in intents or "Inside-Topic" in intents:
                intents.remove("Start-Topic") if "Start-Topic" in intents\
                else intents.remove("Inside-Topic")
            if "Null" in sections:
                sections.remove("Null")
            if "Chitchat" not in intents:
                intent_str = "[Intents]: " + ", ".join(intents) if intents != [] else "[Intents]: "
                section_str = "[Sections]: " + ", ".join(sections) if sections != [] else "[Sections]: "
                return_str = return_str + " " + text + " " + section_str + " " + intent_str
   
        gpt_prompt = [{
            "role": "user",
            "content": GPTEnum.instruct_user.value.replace("$$CONV$$", return_str)
            },{
            "role": "assistant",
            "content": GPTEnum.instruct_assistant.value + candidate_note
            }]
        return gpt_prompt
    
    def prepare_gpt_section_specific_prompts(self, dialogue, filter, in_context):
        dialogue = dialogue
        section_prompts = {}
        for section in self.section_data_dict['clinicalnlp_taskB_test1'].keys():
            messages = []
            messages.append({"role":"developer", "content": self.section_prompts_dict[section]["dev"]})
            out_string = ""
            if filter:
                filtered_text, filtered_intents, filtered_sections = self.dialogue_filter.filter_dialogue(dialogue) 
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
                user_prompt = self.section_prompts_dict[section]["user"].replace("$$CONV$$", out_string) if out_string != "" else ""
            else:
                user_prompt = self.section_prompts_dict[section]["user"].replace("$$CONV$$", dialogue)
            if in_context:
                if user_prompt != "":
                    candidate_messages = self.create_section_specific_bm25_candidates(dialogue, section, filter)
                
                for candidate_input in candidate_messages:
                    messages.append(candidate_input)
            
            messages.append({"role":"user", "content": user_prompt})
            messages.append({"role":"assistant", "content": self.section_prompts_dict[section]["assistant"]})
            section_prompts[section] = messages
        return section_prompts
    
    def prepare_gpt_filtered_prompt(self, dialogue, in_context):
        messages = []
        messages.append({"role":"developer", "content":self.prompt_enum.instruct_prompt_developer.value})
        filtered_text, filtered_intents, filtered_sections = self.dialogue_filter.filter_dialogue(dialogue)
        filtered_text = self.append_outputs_to_text(filtered_sections, filtered_text, "section")
        filtered_text = self.append_outputs_to_text(filtered_intents, filtered_text, "intent")
        filtered_text = " ".join(filtered_text)
        if in_context:
            scores = self.full_note_bm25.get_scores(dialogue.split(" "))
            candidates  = self.candidate_data.loc[np.argpartition(scores, -self.top_k)[-self.top_k:]]
            candidates["prompt_example"] = candidates.apply(lambda x: self.prepare_gpt_candidates(x), axis=1)
            candidate_list = [x for xs in candidates["prompt_example"].to_list() for x in xs]
            for candidate_input in candidate_list:
                messages.append(candidate_input)
        messages.append({"role":"user", "content": self.prompt_enum.instruct_user.value.replace("$$CONV$$", filtered_text)})
        messages.append({"role":"assistant", "content":self.prompt_enum.instruct_assistant.value})
        return messages
    
    def infer_gpt(self, messages, sections):
        if sections:
            out_dict = {}
            for section, message in messages.items():
                if message != "":
                    completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=message,
                    temperature=0.2,
                    max_completion_tokens=256
                )
                    out_dict[section] = completion.choices[0].message.content 
                else:
                    out_dict[section] = "#####EMPTY#####"
            return out_dict
        else:
            
            completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=512
                )

            return completion.choices[0].message.content     
def generate_section_summaries(data, client):
    out_dict = {}
    for section, prompt in data.items():
        if prompt != "":
            out_dict[section] = client.text_generation(prompt=prompt, max_new_tokens=128, temperature=0.2)
        else:
            out_dict[section] = "#####EMPTY#####"
    return out_dict

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





