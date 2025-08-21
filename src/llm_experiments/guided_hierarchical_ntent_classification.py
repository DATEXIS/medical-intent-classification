from pydantic import BaseModel
from outlines import models, generate
#from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from enum import Enum
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
from ast import literal_eval
class Intents(BaseModel):
    acute_assessment: bool
    acute_symptoms: bool
    chitchat: bool
    diagnostic_testing: bool
    discussion: bool
    drug_history: bool
    family_history: bool
    follow_up: bool
    greetings: bool
    lab_examination: bool
    medication: bool
    other_socials: bool
    other_treatments: bool
    personal_history: bool
    physical_examination: bool
    radiology_examination: bool
    reassessment: bool
    referral: bool
    therapeutic_history: bool
    vegetative_history: bool
class SubjectiveIntents(BaseModel):   
    acute_symptoms: bool
    drug_history: bool
    family_history: bool
    greetings: bool
    other_socials: bool
    personal_history: bool
    therapeutic_history: bool
    vegetative_history: bool
class ObjectiveIntents(BaseModel):
    lab_examination: bool
    physical_examination: bool
    radiology_examination: bool
class AssessmentIntents(BaseModel):
    acute_assessment: bool
    reassessment: bool 
class PlanIntents(BaseModel):
    diagnostic_testing: bool
    referral: bool
    other_treatments: bool
    follow_up: bool
    medication: bool
    discussion: bool
class Sections(BaseModel):
    assessment: bool
    null: bool
    objective: bool
    plan: bool
    subjective: bool

class PromptEnum(str, Enum):
    subj_intent_system = "You are a medical AI assistant that classifies doctor utterances into subjective medical intents. You will receive an explanation for all possible intents. \
There are 8 subjective doctor intents available. Each utterance can contain multiple intents. Respond with true or false for all intents if it is present in the utterance or not. The following is a short description for each intent.\
(intent 1) Acute Symptoms: In an “acute symptom(s)” intent a doctor assesses the current symptoms of the patient. This intent is characterized by direct questions about the symptomatic or follow-up questions about symptoms the patient describes. It is one of the more common intents and usually present at the start of a conversation.\
(intent 2) Drug History: The “drug history” intent captures the consumption of drugs by the patient currently or in the past. Everything besides regulated medication we consider as drugs. This includes alcohol, caffeine, nicotine, cannabis and all other “harder” drugs.\
(intent 3) Family History: This intent assesses medical events in the family of the patient. These samples often include direct questions towards similar symptoms in relatives.\
(intent 4) Greetings: If the utterance contains a direct greeting to a patient it is classified as Greetings.\
(intent 5) Other Socials: The “other socials” intent is in usage as an umbrella intent to capture information relating to the current social status a patient inherits or any other factors that affect the patient from the outside. This includes questions towards kids, marriage, job, living situation, social support systems, sports etc.\
(intent 6) Personal History: The “personal history” intent includes all questions related to previous medical events of the patient. This also includes questions directed towards chronic illnesses like diabetes or measuring of the heart rate. This intent additionally captures questions asked about symptoms in the past that might relate to the current complaint. The diet of a patient is also part of the “personal history” intent. \
(intent 7) Therapeutic History: The “therapeutic history”, aka “medication history”, intent occurs when a doctor asks for information regarding medications or therapies the patient applied in the past or is actively consuming.\
(intent 8) Vegetative History: The “vegetative history” intent describes questions towards the internal body functions of a patient. For example it captures instances in which the doctor asks about the fatigueness and general questions regarding the review of system."
    
    obj_intent_system = "You are a medical AI assistant that classifies doctor utterances into objective medical intents. You will receive an explanation for all possible intents. \
There are 3 objective doctor intents available. Each utterance can contain multiple intents. Respond with true or false for all intents if it is present in the utterance or not. The following is a short description for each intent.\
(intent 1) Lab Examination: With the “lab results” intent a doctor is evaluating measurements done in a lab. Indications for this intent is the doctor referring to some external evidence regarding the symptoms of the patient.\
(intent 2) Physical Examination: The “physical examination” intent is the doctor doing physical tests with the patient. This intent is mostly straight-forward, given that the doctor has to ask for permission to do physical tests.\
(intent 3) Radiology Examination: The “radiology results” intent follows the evaluation of screening results for the patient. Screening procedures include x-ray, mrt, echocardiogram etc."

    assessment_intent_system = "You are a medical AI assistant that classifies doctor utterances into diagnoses medical intents. You will receive an explanation for all possible intents. \
There are 2 diagnoses doctor intents available. Each utterance can contain multiple intents. Respond with true or false for all intents if it is present in the utterance or not. The following is a short description for each intent.\
(intent 1) Acute Assessment: The intent of the doctor for the “acute assessment” is to summarize the findings into a conclusive primary diagnosis for the current complaint the patient has. \
(intent 2) Reassessment: The “reassessment” intent captures every assessment towards diagnoses that are not novel for the patient. This is the cases for chronic illnesses and follow-up visits."
    
    plan_intent_system = "You are a medical AI assistant that classifies doctor utterances into plan medical intents. You will receive an explanation for all possible intents. \
There are 6 doctor plan intents available. Each utterance can contain multiple intents. Respond with true or false for all intents if it is present in the utterance or not. The following is a short description for each intent.\
(intent 1) Diagnostic Testing: The “diagnostic testing” intent refers to the doctor ordering any kind of additional medical test to further assess the situation of the patient. This includes lab work (blood results etc.), imagery (mrt, ecg, x-ray etc.) and measuring blood pressure as well.\
(intent 2) Discussion: This intent aims to give the patient an opportunity to ask questions or clarify questions by the patient. It is also about discussing the usage of treatments and their circumstances.\
(intent 3) Follow-up: With this intent the doctor orders a follow-up to check for persistent symptoms or changes due to given medications.\
(intent 4) Medication: In this intent the doctor prescribes the patient the intake of medications.\
(intent 5) Other Treatments: We can not cover all types of treatments specifically. This intent covers those that are not specifically mentioned in this document. This mainly includes prescriptions of orthopedic devices such as crutches or slings. It also covers suggestions towards a better diet and the need of surgery.\
(intent 6) Referral: The “referral” intent is apparent when a doctor plans to refer the patient to another specialist. This can include many professions, like an ophthalmologist or cardiologist, but can also include referral for physical therapy."
    
    chitchat_intent_system = "You are a medical AI assistant that classifies doctor utterances if they contain medical informations or just chitchat. \
Respond with true or false for all intents if the utterance is chitchat or not. We define chitchat as follows.\
(intent 1) Chitchat: If the utterances does not contain any medical intent it is classified as Chitchat."
    intent_user = "Classify the following utterance: $$SAMPLE$$."
    intent_assistant = "Intents: "

    section_system = "You are a medical AI assistant that classifies doctor utterances into medical sections of medical discharge notes. You will receive an explanation for all possible sections. \
There are 5 sections available. Each utterance can belong to multiple sections. Respond with true or false for all sections if it is present in the utterance or not. The following is a short description for each section.\
(section 1) Assessment: This section includes utterances in which the doctor gives the patient information relating to a diagnoses. These diagnosis can be for the current chief complaint or towards chronic illnesses, too. \
(section 2) Null: The Null sections marks utterances which do not contain any medical information relatable to the other sections. It is used for chit-chats between doctor and patients.\
(section 3) Objective: This section includes utterances in which the doctor does a physical examination or discusses laboratory or radiology examinations.\
(section 4) Plan: The Plan section includes utterances that contain information towards the treatment of the patient. Utterances that discusses medication, follow-ups, referrals, surgeries, other treatments and questions are contained in this section.\
(section 5) Subjective: The Subjective section contains utterances that asses the subjective informations of the patient. The doctor asks for symptoms or past medical events or treatments. Also information like family history, drug history or other social situations are dicsusses in this section."
    section_user = "Classify the following utterance: $$SAMPLE$$."
    section_assistant = "Sections: "

def parse_candidate_intents_to_string(candidate, token_dict):
    intents = [x.lower().replace(" ", "_") for x in candidate.intents]
    return_string = token_dict["user"] + PromptEnum.intent_user.value.replace("$$SAMPLE$$" , candidate.text) + token_dict["assistant"] + PromptEnum.intent_assistant.value
    for intent_key in Intents.model_fields.keys():
        if intent_key in intents:
            return_string = return_string + f"{intent_key}=True "
        else:
            return_string = return_string + f"{intent_key}=False "
    
    return return_string + token_dict["eot"]

def parse_candidate_sections_to_string(candidate, token_dict):
    sections = [x.lower() for x in candidate.sections]
    return_string = token_dict["user"] + PromptEnum.section_user.value.replace("$$SAMPLE$$" , candidate.text) + token_dict["assistant"] + PromptEnum.section_assistant.value
    for section_key in Sections.model_fields.keys():
        if section_key in sections:
            return_string = return_string + f"{section_key}=True "
        else:
            return_string = return_string + f"{section_key}=False "
    
    return return_string + token_dict["eot"]

def create_few_shot_candidates(sample, num_candidates, corpus, task, token_dict, system_prompt):
    tokenized_corpus = [doc.split(" ") for doc in corpus.text.to_list()]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(sample.split(" "))
    candidate_ids = np.argpartition(scores, -num_candidates)[-num_candidates:]
    candidates = corpus.iloc[candidate_ids]
    few_shot_examples = " ".join(candidates.apply(lambda x: parse_candidate_intents_to_string(x, token_dict), axis=1).to_list()) if task == "intent"\
    else " ".join(candidates.apply(lambda x: parse_candidate_sections_to_string(x, token_dict), axis=1).to_list())
    full_input_string = few_shot_examples + token_dict["user"] + "Classify the following utterance: " + sample
    if task == "intent":
        return token_dict["system"] + system_prompt + token_dict["eot"] + full_input_string + token_dict["assistant"] + PromptEnum.intent_assistant.value
    else:
        return token_dict["system"] + system_prompt + token_dict["eot"] + full_input_string + token_dict["assistant"] + PromptEnum.section_assistant.value

def prepare_all_prompts(sample, token_dict):
    section_prompt = token_dict["system"] + PromptEnum.section_system.value + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.section_user.value.replace("$$SAMPLE$$", sample) + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.section_assistant.value
    
    subj_prompt = token_dict["system"] + PromptEnum.subj_intent_system.value + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.intent_user.value.replace("$$SAMPLE$$", sample) + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.intent_assistant.value
    
    obj_prompt = token_dict["system"] + PromptEnum.obj_intent_system.value + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.intent_user.value.replace("$$SAMPLE$$", sample) + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.intent_assistant.value

    plan_prompt = token_dict["system"] + PromptEnum.plan_intent_system.value + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.intent_user.value.replace("$$SAMPLE$$", sample) + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.intent_assistant.value
    
    ass_prompt = token_dict["system"] + PromptEnum.assessment_intent_system.value + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.intent_user.value.replace("$$SAMPLE$$", sample) + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.intent_assistant.value

    return pd.Series({"section": section_prompt, "subjective" : subj_prompt, "objective": obj_prompt, "plan": plan_prompt, "assessment":ass_prompt, "text":sample})

def generate_classification(inputs, generator, model_name):
    if "qwen" in model_name.lower():
        inputs_ = []
        for prompt in inputs.to_list():
            inputs_.append(prompt.replace("You are a medical AI assistant", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant"))
    else:
        inputs_ = inputs.to_list()
    print("Batching")
    batched_inputs = list(batch(inputs_, args.batch_size))
    print(f"Success\nBatchsize: {len(batched_inputs[0])}")
    outputs = []
    counter = 0
    for batched in tqdm(batched_inputs, desc="Batch"):
        outputs.append(generator(batched))
        if counter % 10 == 0:
           print(f"Done Batch {counter}")
        counter += 1
    print("Done")
    outputs = pd.Series([x for xs in outputs for x in xs])
    return outputs
    
    
def parse_categories(outputs):
    subj = outputs.subjective
    obj = outputs.objective
    plan = outputs.plan
    assessment = outputs.assessment
    chitchat = outputs.null
    if not (subj and obj and plan and assessment and chitchat):
        chitchat = True
    return pd.Series({"subjective": subj, "obj":obj, "plan":plan, "assessment":assessment, "chitchat":chitchat})

def apply_hierarchy_to_output(sample):
    sections = sample["section"].dict()
    all_false = True
    tmp_out = []
    for category, v in sections.items():
        if category == "null":
            chitchat = v
            continue
        if not v:
            tmp_dict = {}
            for intent in sample[category].dict().keys():
                tmp_dict[intent] = False
            tmp_out.append(tmp_dict)
        else:
            all_false = False
            tmp_out.append(sample[category].dict())
    if chitchat or all_false:
        tmp_out.append({"chitchat":True})
    else:
        tmp_out.append({"chitchat":False})
    return {k: v for d in tmp_out for k, v in d.items()}

        

def batch(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default ="")
    parser.add_argument("--dev_file", type=str,default ="")
    parser.add_argument("--test_file", type=str, default ="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--data_target", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--llm_setting", type=str, default="few_shot")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--task", type=str, default = "intent")
    args = parser.parse_args()
    target_dataset = args.data_target
    print("Outputpath: "  + f"{target_dataset}_{args.llm_setting}_output.csv")
    training_data = pd.read_json(args.train_file)
    validation_data = pd.read_json(args.dev_file)
    test_data = pd.read_json(args.test_file)
    candidate_data = pd.concat((training_data, validation_data)).reset_index(drop=True)
    token_dict = {}
    if "llama" in args.model_name.lower():
        token_dict["system"] = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        token_dict["user"]="<|start_header_id|>user<|end_header_id|>"
        token_dict["assistant"] = "<|start_header_id|>assistant<|end_header_id|>"
        token_dict["eot"]  = "<|eot_id|>"
    elif "qwen" in args.model_name.lower():
        token_dict["system"] = "<|im_start|>system "
        token_dict["user"] ="<|im_start|>user "
        token_dict["assistant"] = "<|im_start|>assistant "
        token_dict["eot"] = "<|im_end|>"
    elif "phi" in args.model_name.lower():
        token_dict["system"] = "<|im_start|>system<|im_sep|>"
        token_dict["user"] ="<|im_start|>user<|im_sep|>"
        token_dict["assistant"] = "<|im_start|>assistant<|im_sep|>"
        token_dict["eot"] = "<|im_end|>"

    inputs = test_data.text.apply(lambda x: prepare_all_prompts(x, token_dict))
    stop = inputs.apply(lambda x: create_few_shot_candidates(x, args.num_candidates, candidate_data, args.task, token_dict, system_prompt))

    
 
    system_prompt = PromptEnum.section_system.value if "qwen" not in args.model_name.lower() \
        else PromptEnum.section_system.value.replace("You are a medical AI assistant", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant")
    prompt = token_dict["system"] + system_prompt + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.section_user.value + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.section_assistant.value
    if args.task == "intent":
        system_prompt = PromptEnum.intent_system.value if "qwen" not in args.model_name.lower() \
            else PromptEnum.intent_system.value.replace("You are a medical AI assistant", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant")
        prompt = token_dict["system"] + system_prompt + token_dict["eot"] + \
                token_dict["user"] + PromptEnum.intent_user.value + token_dict["eot"] + \
                token_dict["assistant"] + PromptEnum.intent_assistant.value
    llm_section_inputs = test_data.text.map(lambda x: prompt.replace("$$SAMPLE$$", x)).to_list()
        

   

    

    print("Load model")          
    model = models.transformers(args.model_name, device="auto")
    print("Create generator for section classification")
    generators = {
        "section":generate.json(model, Sections),
        "subjective": generate.json(model, SubjectiveIntents),
        "objective" : generate.json(model, ObjectiveIntents),
        "plan": generate.json(model, PlanIntents),
        "assessment" : generate.json(model, AssessmentIntents)
    } 

    
    outputs = {}
    for k,generator in generators.items():
        print(f"Start generating {k}")
        outputs[k] = generate_classification(inputs[k], generator, args.model_name)
    intent_categories = outputs["section"].apply(parse_categories)
    out_df = pd.DataFrame.from_dict(outputs).apply(apply_hierarchy_to_output, axis=1)
    print("Saving")

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    out_df.to_csv(args.output_path + f"{target_dataset}_{args.llm_setting}_intents_output.csv", index=False)
    outputs["section"].to_csv(args.output_path + f"{target_dataset}_{args.llm_setting}_sections_output.csv", index=False)
    
