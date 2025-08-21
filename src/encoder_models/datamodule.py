import json
import random
from typing import Optional
import csv
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertTokenizerFast
import numpy as np
import ast
import pandas as pd
import pickle



class MultiTaskClassificationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        text = [x["text"] for x in data]
        
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
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

          
        intent_labels = torch.stack([x["intent_labels"] for x in data])
        section_labels = torch.stack([x["section_labels"] for x in data])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "intent_labels": intent_labels,
            "section_labels": section_labels,
        }
       
class MutliTaskClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, intent_label_lookup, section_label_lookup):
        self.examples = examples
        self.intent_label_lookup = intent_label_lookup
        self.section_label_lookup = section_label_lookup
        self.inverse_intent_label_lookup = {v: k for k, v in intent_label_lookup.items()}
        self.inverse_section_label_lookup = {v: k for k, v in section_label_lookup.items()}
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples.loc[idx]
        text = example["text"]
        intent_labels = example["intents"]
        intent_label_ids = [self.intent_label_lookup[x] for x in intent_labels]
        intent_label_idxs = torch.tensor([int(x) for x in intent_label_ids])
        intent_labels = torch.zeros(len(self.intent_label_lookup), dtype=torch.float32)
        if len(intent_label_idxs) != 0:
            intent_labels[intent_label_idxs] = 1
        section_labels = example["sections"]
        section_label_ids = [self.section_label_lookup[x] for x in section_labels]
        section_label_idxs = torch.tensor([int(x) for x in section_label_ids])
        section_labels = torch.zeros(len(self.section_label_lookup), dtype=torch.float32)
        if len(section_label_idxs) != 0:
            section_labels[section_label_idxs] = 1
        return {"text": text, "intent_labels":intent_labels, "section_labels":section_labels}
    

class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "",
        train_file: str = None,
        dev_file: str = None,
        test_file: str = None,
        batch_size: int = 32,
        eval_batch_size: int = 16,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_workers: int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if train_file:
            training_data = pd.read_json(train_file)
            validation_data = pd.read_json(dev_file)
            test_data = pd.read_json(test_file)
        else:
            data = pd.read_json(data_path)
            training_data = data.loc[data["split"] == "train"].drop("split", axis=1).reset_index(drop=True)
            validation_data = data.loc[data["split"] == "val"].drop("split", axis=1).reset_index(drop=True)
            test_data = data.loc[data["split"] == "test1"].drop("split", axis=1).reset_index(drop=True)
        self.all_intent_labels = sorted(training_data.intents.explode().dropna().unique().tolist())
        self.all_section_labels = sorted(training_data.sections.explode().dropna().unique().tolist())
              
        intent_label_idx = {v: k for k, v in enumerate(self.all_intent_labels)}
        section_label_idx = {v: k for k, v in enumerate(self.all_section_labels)}
        
        self.training_data = training_data[
                ["text", "intents", "sections"]
            ].reset_index(drop=True)

        self.test_data = test_data[
                ["text", "intents", "sections"]
            ].reset_index(drop=True) 
        self.val_data = validation_data[
                ["text", "intents", "sections"]
            ].reset_index(drop=True)
               # build label index
        self.intent_label_idx = intent_label_idx
        self.section_label_idx = section_label_idx
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collator = MultiTaskClassificationCollator(self.tokenizer, max_seq_len)
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        train = MutliTaskClassificationDataset(
            self.training_data,
            intent_label_lookup=self.intent_label_idx,
            section_label_lookup=self.section_label_idx,
        )


        val = MutliTaskClassificationDataset(
            self.val_data,
            intent_label_lookup=self.intent_label_idx,
            section_label_lookup=self.section_label_idx,
        )

        test = MutliTaskClassificationDataset(
            self.test_data,
            intent_label_lookup=self.intent_label_idx,
            section_label_lookup=self.section_label_idx,
        )
    
        self.train = train
        self.val = val
        self.test = test
        print("Val length: ", len(self.val))
        print("Train Length: ", len(self.train))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
    

class NextIntentClassificationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        text = [x["src"] for x in data]
        self.tokenizer.truncation_side = "left"
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
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

          
        intent_labels = torch.stack([x["intent_labels"] for x in data])
        section_labels = torch.stack([x["section_labels"] for x in data])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "intent_labels": intent_labels,
            "section_labels": section_labels,
            "tokenizer_len": len(self.tokenizer)
        }
class NextIntentClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, intent_label_lookup, section_label_lookup):
        self.examples = examples
        self.intent_label_lookup = intent_label_lookup
        self.section_label_lookup = section_label_lookup
        self.inverse_intent_label_lookup = {v: k for k, v in intent_label_lookup.items()}
        self.inverse_section_label_lookup = {v: k for k, v in section_label_lookup.items()}
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples.loc[idx]
        text = example["src"]
        intent_labels = example["intents"]
        intent_label_ids = [self.intent_label_lookup[x] for x in intent_labels]
        intent_label_idxs = torch.tensor([int(x) for x in intent_label_ids])
        intent_labels = torch.zeros(len(self.intent_label_lookup), dtype=torch.float32)
        if len(intent_label_idxs) != 0:
            intent_labels[intent_label_idxs] = 1
        section_labels = example["sections"]
        section_label_ids = [self.section_label_lookup[x] for x in section_labels]
        section_label_idxs = torch.tensor([int(x) for x in section_label_ids])
        section_labels = torch.zeros(len(self.section_label_lookup), dtype=torch.float32)
        if len(section_label_idxs) != 0:
            section_labels[section_label_idxs] = 1

        return {"src": text, "intent_labels":intent_labels, "section_labels":section_labels}
    

class NextIntentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "",
        train_file: str = None,
        dev_file: str = None,
        test_file: str = None,
        batch_size: int = 32,
        eval_batch_size: int = 16,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_workers: int = 8,
        max_seq_len: int = 512,
        add_class_tokens: bool = False
    ):
        super().__init__()
        if train_file:
            training_data = pd.read_json(train_file)
            validation_data = pd.read_json(dev_file)
            test_data = pd.read_json(test_file)
        else:
            data = pd.read_json(data_path)
            training_data = data.loc[data["split"] == "train"].drop("split", axis=1).reset_index(drop=True)
            validation_data = data.loc[data["split"] == "val"].drop("split", axis=1).reset_index(drop=True)
            test_data = data.loc[data["split"] == "test1"].drop("split", axis=1).reset_index(drop=True)
        self.all_intent_labels = sorted(training_data.intents.explode().dropna().unique().tolist())
        self.all_section_labels = sorted(training_data.sections.explode().dropna().unique().tolist())
              
        self.intent_label_idx = {v: k for k, v in enumerate(self.all_intent_labels)}
        self.section_label_idx = {v: k for k, v in enumerate(self.all_section_labels)}
        
        self.training_data = training_data[
                ["src", "intents","sections"]
            ].reset_index(drop=True)

        self.test_data = test_data[
                ["src", "intents","sections",]
            ].reset_index(drop=True) 
        self.val_data = validation_data[
                ["src", "intents","sections"]
            ].reset_index(drop=True)

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if add_class_tokens:
            class_labels = ["["+name+"]" for name in self.training_data.intents.explode().value_counts().to_dict().keys()]
            class_labels.append("[Doctor]")
            class_labels.append("[Patient]")
            class_labels.append("[Patient Guest]")
            class_labels.append("[Conversation Start]")
            self.tokenizer.add_special_tokens({"additional_special_tokens": class_labels})
        self.collator = NextIntentClassificationCollator(self.tokenizer, max_seq_len)
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        train = NextIntentClassificationDataset(
            self.training_data,
            intent_label_lookup=self.intent_label_idx,
            section_label_lookup=self.section_label_idx,
        )


        val = NextIntentClassificationDataset(
            self.val_data,
            intent_label_lookup=self.intent_label_idx,
            section_label_lookup=self.section_label_idx,
        )

        test = NextIntentClassificationDataset(
            self.test_data,
            intent_label_lookup=self.intent_label_idx,
            section_label_lookup=self.section_label_idx,
        )
    
        self.train = train
        self.val = val
        self.test = test
        print("Val length: ", len(self.val))
        print("Train Length: ", len(self.train))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )