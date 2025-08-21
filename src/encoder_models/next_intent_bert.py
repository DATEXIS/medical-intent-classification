from typing import Optional, Dict, Any, List, Union
import os
import lightning.pytorch as pl
import torch
import torchmetrics
import pandas as pd
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelRecall, MultilabelPrecision
import transformers
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertModel, AutoModel
import torch.nn.functional as F
from torchmetrics.functional import f1_score, precision, recall, accuracy, auroc, average_precision
from pathlib import Path


class NextIntentBertClassificationModel(pl.LightningModule):
    def __init__(self,
                 tasks: dict = {},
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 num_training_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 device: str = "cpu",
                 num_classes: int = 20,
                 save_scores: bool = False,
                 hidden_dim=1024,
                 eval_treshold: float = 0.25,
                 test_out: str = "",
                 is_test: bool = False,
                 embedding_dim: int = 30522
                 ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        if is_test:
            if self.encoder.get_input_embeddings().weight.shape[0] != embedding_dim:
                with torch.cuda.amp.autocast(enabled=False):
                    self.encoder.to(torch.float32)
                    self.encoder.resize_token_embeddings(embedding_dim)
                    #self.encoder.to(torch.bfloat16)
            self.classification_layer = self.classification_layer.to(torch.float32)
        else:
            self.classification_layer = self.classification_layer.to(torch.bfloat16)
        self.num_classes = num_classes

        self.classification_layer = torch.nn.Linear(hidden_dim, self.num_classes)
        
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.val_output = []
        self.test_output = []
        self.save_scores = save_scores
        self.eval_treshold = eval_treshold
        self.test_out = test_out
    def forward(self,
                batch):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]

        logits = self.classification_layer(encoded)
        probs = torch.sigmoid(logits)
        return probs

    def training_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
 
        logits = self.classification_layer(encoded)
        labels = batch["intent_labels"]
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("Train/Loss", loss)

        return loss


    def test_step(self, batch, batch_idx, **kwargs):
        if self.encoder.get_input_embeddings().weight.shape[0] != batch["tokenizer_len"]:
            with torch.cuda.amp.autocast(enabled=False):
                self.encoder.to(torch.float32)
                self.encoder.resize_token_embeddings(batch["tokenizer_len"])
                self.encoder.to(torch.bfloat16)
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        self.test_output.append({"logits": logits, "labels": batch["intent_labels"]})
        return {"logits": logits, "labels": batch["intent_labels"]}

    def on_test_epoch_end(self) -> None:


        logits = torch.cat([x["logits"] for x in self.test_output])
        labels = torch.cat([x["labels"] for x in self.test_output]).int() 
        y_pred = torch.sigmoid(logits)
        
        preds = y_pred

        macro_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")
        micro_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="micro")
        all_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="none")


        macro_score_auroc = auroc(y_pred, labels, 'multilabel', num_labels=self.num_classes, average="macro")
        micro_score_auroc = auroc(y_pred, labels, 'multilabel', num_labels=self.num_classes, average="micro")   
        all_score_auroc = auroc(y_pred, labels, 'multilabel', num_labels=self.num_classes, average="none")

        macro_score_ap = average_precision(y_pred, labels, 'multilabel', num_labels=self.num_classes, average="macro")
        micro_score_ap = average_precision(y_pred, labels, 'multilabel', num_labels=self.num_classes, average="micro")   
        all_score_ap = average_precision(y_pred, labels, 'multilabel', num_labels=self.num_classes, average="none")

        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        self.log(f"Test/F1_macro", macro_score_f1, sync_dist=True)
        self.log(f"Test/Auroc_macro", macro_score_auroc, sync_dist=True)
        self.log(f"Test/AP_macro", macro_score_ap, sync_dist=True)
        self.log(f"Test/Loss", loss, sync_dist=True)
        results = {"f1":{"macro":macro_score_f1.tolist(), "micro":micro_score_f1.tolist(), "all": all_score_f1.tolist()},\
                                    #"recall":{"macro":macro_score_recall.tolist(), "micro":micro_score_recall.tolist(), "all": all_score_recall.tolist()},\
                                    #"precision":{"macro":macro_score_precision.tolist(), "micro":micro_score_precision.tolist(), "all": all_score_precision.tolist()},\
                                        #"accuracy":{"macro":macro_score_accuracy.tolist(), "micro":micro_score_accuracy.tolist(), "all": all_score_accuracy.tolist()},\
                                        "auroc":{"macro":macro_score_auroc.tolist(), "micro":micro_score_auroc.tolist(), "all": all_score_auroc.tolist()},\
                                            "accuracy":{"macro":macro_score_ap.tolist(), "micro":micro_score_ap.tolist(), "all": all_score_ap.tolist()}}

        if self.save_scores:
            experiment_dir = self._trainer.log_dir.split("lightning")[0]
            experiment_df = pd.DataFrame.from_dict(results)
            Path(self.test_out).mkdir(exist_ok=True)
            experiment_df.to_json(f"{self.test_out}/next_intent_test_results.json")

        self.test_output = list()


    def validation_step(self, batch, batch_idx,**kwargs):
        if self.encoder.get_input_embeddings().weight.shape[0] != batch["tokenizer_len"]:
            with torch.cuda.amp.autocast(enabled=False):
                self.encoder.to(torch.float32)
                self.encoder.resize_token_embeddings(batch["tokenizer_len"])
                self.encoder.to(torch.bfloat16)
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)        
        self.val_output.append({"logits": logits, "labels": batch["intent_labels"]})
        return {"logits": logits, "labels": batch["intent_labels"]}
        
    
    def on_validation_epoch_end(self) -> None:

        logits = torch.cat([x["logits"] for x in self.val_output])
        labels = torch.cat([x["labels"] for x in self.val_output]).int() 
        preds = torch.sigmoid(logits)
        score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")\
                
        score_auroc = auroc(preds, labels, 'multilabel', num_labels=self.num_classes, average="macro")
        
        score_ap = average_precision(preds, labels, 'multilabel', num_labels=self.num_classes, average="macro")
                
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        self.log(f"Val/F1", score_f1, sync_dist=True)
        self.log(f"Val/Auroc", score_auroc, sync_dist=True)
        self.log(f"Val/AP", score_ap, sync_dist=True)
        self.log(f"Val/Loss", loss, sync_dist=True)
        experiment_dir = self._trainer.log_dir.split("lightning")[0]
        self.val_output = list()
        
    
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
    def eval(self):
        self.encoder.eval()
        self.classification_layer.eval()


