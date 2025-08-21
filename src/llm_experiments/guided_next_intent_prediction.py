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


class PromptEnum(str, Enum):
    intent_system = "You are a medical AI assistant that predicts the next medical intents in a doctor-patient interview. You'll be given up to 5 previous doctor-patient interactions and you have to predict what medical intent the doctor has for the next interaction.\
There are 20 doctor intents available. An interaction can have multiple medical intents. Respond with true or false for all intents if it is present for the next interaction. The following is a short description for each intent.\
(intent 1) Acute Assessment: The intent of the doctor for the “acute assessment” is to summarize the findings into a conclusive primary diagnosis for the current complaint the patient has. \
(intent 2) Acute Symptoms: In an “acute symptom(s)” intent a doctor assesses the current symptoms of the patient. This intent is characterized by direct questions about the symptomatic or follow-up questions about symptoms the patient describes. It is one of the more common intents and usually present at the start of a conversation.\
(intent 3) Chitchat: If the utterances does not contain any medical intent it is classified as Chitchat.\
(intent 4) Diagnostic Testing: The “diagnostic testing” intent refers to the doctor ordering any kind of additional medical test to further assess the situation of the patient. This includes lab work (blood results etc.), imagery (mrt, ecg, x-ray etc.) and measuring blood pressure as well.\
(intent 5) Discussion: This intent aims to give the patient an opportunity to ask questions or clarify questions by the patient. It is also about discussing the usage of treatments and their circumstances.\
(intent 6) Drug History: The “drug history” intent captures the consumption of drugs by the patient currently or in the past. Everything besides regulated medication we consider as drugs. This includes alcohol, caffeine, nicotine, cannabis and all other “harder” drugs.\
(intent 7) Family History: This intent assesses medical events in the family of the patient. These samples often include direct questions towards similar symptoms in relatives.\
(intent 8) Follow-up: With this intent the doctor orders a follow-up to check for persistent symptoms or changes due to given medications.\
(intent 9) Greetings: If the utterance contains a direct greeting to a patient it is classified as Greetings.\
(intent 10) Lab Examination: With the “lab results” intent a doctor is evaluating measurements done in a lab. Indications for this intent is the doctor referring to some external evidence regarding the symptoms of the patient.\
(intent 11) Medication: In this intent the doctor prescribes the patient the intake of medications.\
(intent 12) Other Socials: The “other socials” intent is in usage as an umbrella intent to capture information relating to the current social status a patient inherits or any other factors that affect the patient from the outside. This includes questions towards kids, marriage, job, living situation, social support systems, sports etc.\
(intent 13) Other Treatments: We can not cover all types of treatments specifically. This intent covers those that are not specifically mentioned in this document. This mainly includes prescriptions of orthopedic devices such as crutches or slings. It also covers suggestions towards a better diet and the need of surgery.\
(intent 14) Personal History: The “personal history” intent includes all questions related to previous medical events of the patient. This also includes questions directed towards chronic illnesses like diabetes or measuring of the heart rate. This intent additionally captures questions asked about symptoms in the past that might relate to the current complaint. The diet of a patient is also part of the “personal history” intent. \
(intent 15) Physical Examination: The “physical examination” intent is the doctor doing physical tests with the patient. This intent is mostly straight-forward, given that the doctor has to ask for permission to do physical tests.\
(intent 16) Radiology Examination: The “radiology results” intent follows the evaluation of screening results for the patient. Screening procedures include x-ray, mrt, echocardiogram etc. \
(intent 17) Reassessment: The “reassessment” intent captures every assessment towards diagnoses that are not novel for the patient. This is the cases for chronic illnesses and follow-up visits.\
(intent 18) Referral: The “referral” intent is apparent when a doctor plans to refer the patient to another specialist. This can include many professions, like an ophthalmologist or cardiologist, but can also include referral for physical therapy.\
(intent 19) Therapeutic History: The “therapeutic history”, aka “medication history”, intent occurs when a doctor asks for information regarding medications or therapies the patient applied in the past or is actively consuming.\
(intent 20) Vegetative History: The “vegetative history” intent describes questions towards the internal body functions of a patient. For example it captures instances in which the doctor asks about the fatigueness and general questions regarding the review of system."
    intent_user = "Predict the next medical intent given the following sequential doctor-patient interactions: $$SAMPLE$$."
    intent_assistant = "Next Intents: "


def parse_candidate_intents_to_string(candidate, token_dict):
    intents = [x.lower().replace(" ", "_") for x in candidate.intents]
    return_string = token_dict["user"] + PromptEnum.intent_user.value.replace("$$SAMPLE$$" , candidate.src) + token_dict["assistant"] + PromptEnum.intent_assistant.value
    for intent_key in Intents.model_fields.keys():
        if intent_key in intents:
            return_string = return_string + f"{intent_key}=True "
        else:
            return_string = return_string + f"{intent_key}=False "
    
    return return_string + token_dict["eot"]


def create_few_shot_candidates(sample, num_candidates, corpus, token_dict, system_prompt):
    tokenized_corpus = [doc.split(" ") for doc in corpus.src.to_list()]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(sample.split(" "))
    candidate_ids = np.argpartition(scores, -num_candidates)[-num_candidates:]
    candidates = corpus.iloc[candidate_ids]
    few_shot_examples = " ".join(candidates.apply(lambda x: parse_candidate_intents_to_string(x, token_dict), axis=1).to_list()) 
    full_input_string = few_shot_examples + token_dict["user"] + "Predict the next medical intent given the following sequential doctor-patient interactions: " + sample

    return token_dict["system"] + system_prompt + token_dict["eot"] + full_input_string + token_dict["assistant"] + PromptEnum.intent_assistant.value
   

def batch(input_list, batch_size):
  for i in range(0, len(input_list), batch_size):
    yield input_list[i:i + batch_size]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_file", type=str, default ="/Users/toroe/Workspace/medical_intent_verification/dataset/train_next_i_pred_full_5_seq.json")
    parser.add_argument("--dev_file", type=str,default ="/Users/toroe/Workspace/medical_intent_verification/dataset/dev_next_i_pred_full_5_seq.json")
    parser.add_argument("--test_file", type=str, default ="/Users/toroe/Workspace/medical_intent_verification/dataset/test_next_i_pred_full_5_seq.json")
    parser.add_argument("--data_path", type=str, default="/pvc/data/intent_classification/dataset")
    parser.add_argument("--data_target", type=str, default="doc_only")
    parser.add_argument("--output_path", type=str, default="/pvc/experiments/llama31_tst/")
    parser.add_argument("--llm_setting", type=str, default="few_shot")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
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

    system_prompt = PromptEnum.intent_system.value if "qwen" not in args.model_name.lower() \
        else PromptEnum.intent_system.value.replace("You are a medical AI assistant", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant")
    prompt = token_dict["system"] + system_prompt + token_dict["eot"] + \
            token_dict["user"] + PromptEnum.intent_user.value + token_dict["eot"] + \
            token_dict["assistant"] + PromptEnum.intent_assistant.value

    if args.llm_setting == "zero_shot":   
        llm_inputs = test_data.src.map(lambda x: prompt.replace("$$SAMPLE$$", x)).to_list()
    elif args.llm_setting == "few_shot":
        llm_inputs = test_data.src.map(lambda x: create_few_shot_candidates(x, args.num_candidates, candidate_data, token_dict, system_prompt)).to_list()


    print("Batching")
    batched_llm_inputs = list(batch(llm_inputs, args.batch_size))
    print(f"Success\nBatchsize: {len(batched_llm_inputs[0])}")

    print("Load model")          
    model = models.transformers(args.model_name, device="auto", model_kwargs={"token":"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"}, tokenizer_kwargs={"token":"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"})# , "rope_scaling":{"type":"llama3", "factor":32.0}}
    print("Create generator")
    generator = generate.json(model, Intents)

    #output = generator(PromptEnum.zero_shot_llama.value.replace("$$SAMPLE$$", sample_text))
    print("Start generating")
    outputs = []
    counter = 0
    for batched in tqdm(batched_llm_inputs, desc="Batch"):
        outputs.append(generator(batched))
        if counter % 10 == 0:
           print(f"Done Batch {counter}")
        counter += 1
    print("Done")
    outputs = [x for xs in outputs for x in xs]

    print("Saving")
    test_data["prompts"] = test_data.src.map(lambda x: prompt.replace("$$SAMPLE$$", x))
    test_data["outputs"] = pd.Series(outputs)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    test_data.to_csv(args.output_path + f"{target_dataset}_{args.llm_setting}_output.csv", index=False)
    
