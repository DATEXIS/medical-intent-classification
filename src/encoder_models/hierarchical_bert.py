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


class HierarchicalBertClassificationModel(pl.LightningModule):
    def __init__(self,
                 tasks: dict = {},
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 num_training_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 device: str = "mps",
                 gradnorm_alpha: float = 1.0,
                 use_loss_weighting: bool = False,
                 save_scores: bool = False,
                 hidden_dim=1024,
                 eval_treshold: float = 0.25,
                 test_out: str = "",
                 is_test: bool = False,
                 embedding_dim: int = 12345,
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
            self.classification_layers = torch.nn.ModuleDict({k:torch.nn.Linear(hidden_dim, v, device=device).to(torch.float32) for k,v in tasks.items()})
        else:
            self.classification_layers = torch.nn.ModuleDict({k:torch.nn.Linear(hidden_dim, v, device=device).to(torch.bfloat16) for k,v in tasks.items()})
        self.tasks = tasks
        
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.section_to_intent_mapping = [0,4,1,3,3,4,4,3,4,2,3,4,3,4,2,2,0,3,4,4]
        if use_loss_weighting:
            self.task_weights = torch.nn.Parameter(torch.ones(len(self.classification_layers)))
        self.use_loss_weighting = use_loss_weighting
        self.val_output = []
        self.test_output = []
        self.save_scores = save_scores
        self.eval_treshold = eval_treshold
        self.test_out = test_out
    def forward(self,
                batch):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]

        task_logits = {}
        for task, num_classes in self.tasks.items():
            logits = self.classification_layers[task](encoded)
            task_logits[task] = logits

        return task_logits

    def training_step(self, batch, batch_idx, **kwargs):
        epsilon = 1e-6
        task_logits = self(batch)
        losses = []
        loss_weights = []
        task_probs = {}

        for task, num_classes in self.tasks.items():
            probs = torch.sigmoid(task_logits[task])
            task_probs[task] = probs
        intent_mask = torch.stack([
            task_probs["section"][:, parent_idx] > 0.5 for parent_idx in self.section_to_intent_mapping
            ], dim=1).float()
        
        for task, num_classes in self.tasks.items():
            labels = batch[f"{task}_labels"]
            if task == "intent":
                tmp_loss = F.binary_cross_entropy_with_logits(task_logits[task], labels.float(), reduction="none")
                tmp_loss *= intent_mask
                tmp_loss = tmp_loss.sum() / intent_mask.sum().clamp(min=1.0)
            else:
                tmp_loss = F.binary_cross_entropy_with_logits(task_logits[task], labels.float())
            losses.append(tmp_loss)
            if self.use_loss_weighting:
                loss_weights.append(1 / (tmp_loss.item()+ epsilon))
       
        #Normalize weights

        if self.use_loss_weighting:
            total_weight = sum(losses)
            loss_weights = [x / total_weight for x in loss_weights]

            total_loss = losses[0] * loss_weights[0] +\
                losses[1] * loss_weights[1]
        else:
            total_loss = sum(losses) / len(losses)
        self.log("Train/Loss", total_loss)

        return total_loss


    def test_step(self, batch, batch_idx, **kwargs):
        task_logits = self(batch)

        test_dict = {}
        for task_name, num_classes in self.tasks.items():
            test_dict[task_name] = {"logits": task_logits[task_name], "labels": batch["intent_labels"]} if task_name == "intent"\
                else {"logits": task_logits[task_name], "labels": batch["section_labels"]}
            
        self.test_output.append(test_dict)
        return {"logits": task_logits[task_name], "labels": batch["intent_labels"]}

    def on_test_epoch_end(self) -> None:
        losses = []
        loss_weights = []
        epsilon = 1e-6
        task_results = {}
        task_probs = {}

        for task, num_classes in self.tasks.items():
            logits = torch.cat([x[task]["logits"] for x in self.test_output])
            labels = torch.cat([x[task]["labels"] for x in self.test_output]).int()
            probs = torch.sigmoid(logits)
            task_probs[task] = probs
        intent_mask = torch.stack([
            task_probs["section"][:, parent_idx] > 0.5 for parent_idx in self.section_to_intent_mapping
            ], dim=1).float()
        task_probs["intent"] *= intent_mask

        for task_name, num_classes in self.tasks.items():
            logits = torch.cat([x[task_name]["logits"] for x in self.test_output])
            labels = torch.cat([x[task_name]["labels"] for x in self.test_output]).int()
            preds = task_probs[task_name]
            macro_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")
            micro_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="micro")
            all_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="none")

            macro_score_recall = recall(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")
            micro_score_recall = recall(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="micro")
            all_score_recall = recall(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="none")

            macro_score_precision = precision(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")
            micro_score_precision = precision(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="micro")
            all_score_precision = precision(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="none")

            macro_score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")
            micro_score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="micro")   
            all_score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="none")           
            
            macro_score_auroc = auroc(preds, labels, 'multilabel', num_labels=num_classes, average="macro")
            micro_score_auroc = auroc(preds, labels, 'multilabel', num_labels=num_classes, average="micro")   
            all_score_auroc = auroc(preds, labels, 'multilabel', num_labels=num_classes, average="none")

            macro_score_ap = average_precision(preds, labels, 'multilabel', num_labels=num_classes, average="macro")
            micro_score_ap = average_precision(preds, labels, 'multilabel', num_labels=num_classes, average="micro")   
            all_score_ap = average_precision(preds, labels, 'multilabel', num_labels=num_classes, average="none")
            if task_name == "intent":
                tmp_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                tmp_loss *= intent_mask
                tmp_loss = tmp_loss.sum() / intent_mask.sum().clamp(min=1.0)
            else:
                tmp_loss =F.binary_cross_entropy_with_logits(logits, labels.float())
            if self.use_loss_weighting:
                loss_weights.append(1 / (tmp_loss.item()+ epsilon))
            
            self.log(f"Test/{task_name}_F1_macro", macro_score_f1, sync_dist=True)
            self.log(f"Test/{task_name}_Precision_macro", macro_score_precision, sync_dist=True)
            self.log(f"Test/{task_name}_Recall_macro", macro_score_recall, sync_dist=True)
            self.log(f"Test/{task_name}_Accuracy_macro", macro_score_accuracy, sync_dist=True)
            self.log(f"Test/{task_name}_Auroc_macro", macro_score_auroc, sync_dist=True)
            self.log(f"Test/{task_name}_AP_macro", macro_score_ap, sync_dist=True)
            self.log(f"Test/{task_name}_Loss", tmp_loss, sync_dist=True)
            losses.append(tmp_loss)
            task_results[task_name] = {"f1":{"macro":macro_score_f1.tolist(), "micro":micro_score_f1.tolist(), "all": all_score_f1.tolist()},\
                                       #"recall":{"macro":macro_score_recall.tolist(), "micro":micro_score_recall.tolist(), "all": all_score_recall.tolist()},\
                                        #"precision":{"macro":macro_score_precision.tolist(), "micro":micro_score_precision.tolist(), "all": all_score_precision.tolist()},\
                                         #"accuracy":{"macro":macro_score_accuracy.tolist(), "micro":micro_score_accuracy.tolist(), "all": all_score_accuracy.tolist()},\
                                            "auroc":{"macro":macro_score_auroc.tolist(), "micro":micro_score_auroc.tolist(), "all": all_score_auroc.tolist()},\
                                                "accuracy":{"macro":macro_score_ap.tolist(), "micro":micro_score_ap.tolist(), "all": all_score_ap.tolist()}}
        if self.use_loss_weighting:
            total_weight = sum(losses)
            loss_weights = [x / total_weight for x in loss_weights]

            total_loss = losses[0] * loss_weights[0] +\
                losses[1] * loss_weights[1]
        else:
            total_loss = sum(losses) / len(losses)
        if self.save_scores:
            experiment_dir = self._trainer.log_dir.split("lightning")[0]
            experiment_df = pd.DataFrame.from_dict(task_results)
            Path(self.test_out).mkdir(exist_ok=True)
            experiment_df["intent"].to_json(f"{self.test_out}/intent_test_results.json")


        
        self.log("Test/Weighted_loss", total_loss)
        self.test_output = list()


    def validation_step(self, batch, batch_idx,**kwargs):
        if self.encoder.get_input_embeddings().weight.shape[0] != batch["tokenizer_len"]:
            with torch.cuda.amp.autocast(enabled=False):
                self.encoder.to(torch.float32)
                self.encoder.resize_token_embeddings(batch["tokenizer_len"])
                self.encoder.to(torch.bfloat16)
        task_logits = self(batch)
        val_dict = {}
        for task_name, num_classes in self.tasks.items():
            val_dict[task_name] = {"logits": task_logits[task_name], "labels": batch["intent_labels"]} if task_name == "intent"\
                else {"logits": task_logits[task_name], "labels": batch["section_labels"]}
            
        self.val_output.append(val_dict)
        return {"probs": task_logits[task_name], "labels": batch["intent_labels"]}
        
    
    def on_validation_epoch_end(self) -> None:
        losses = []
        loss_weights = []
        epsilon = 1e-6
        task_probs = {}
        
        for task, num_classes in self.tasks.items():
            logits = torch.cat([x[task]["logits"] for x in self.val_output])
            labels = torch.cat([x[task]["labels"] for x in self.val_output]).int()
            probs = torch.sigmoid(logits)
            task_probs[task] = probs
        intent_mask = torch.stack([
            task_probs["section"][:, parent_idx] > 0.5 for parent_idx in self.section_to_intent_mapping
            ], dim=1).float()
        task_probs["intent"] *= intent_mask

        for task_name, num_classes in self.tasks.items():
            logits = torch.cat([x[task_name]["logits"] for x in self.val_output])
            labels = torch.cat([x[task_name]["labels"] for x in self.val_output]).int() 
            preds = task_probs[task_name]
            score_f1 = f1_score(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")\
                   
            score_recall = recall(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")\
                    
            score_precision = precision(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")\
                    
            score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=num_classes, threshold=self.eval_treshold, average="macro")\
            
            score_auroc = auroc(preds, labels, 'multilabel', num_labels=num_classes, average="macro")
            
            score_ap = average_precision(preds, labels, 'multilabel', num_labels=num_classes, average="macro")
                    
            
            if task_name == "intent":
                tmp_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                tmp_loss *= intent_mask
                tmp_loss = tmp_loss.sum() / intent_mask.sum().clamp(min=1.0)
            else:
                tmp_loss =F.binary_cross_entropy_with_logits(logits, labels.float())
            if self.use_loss_weighting:
                loss_weights.append(1 / (tmp_loss.item()+ epsilon))
            
            self.log(f"Val/{task_name}_F1", score_f1, sync_dist=True)
            self.log(f"Val/{task_name}_Precision", score_precision, sync_dist=True)
            self.log(f"Val/{task_name}_Recall", score_recall, sync_dist=True)
            self.log(f"Val/{task_name}_Accuracy", score_accuracy, sync_dist=True)
            self.log(f"Val/{task_name}_Auroc", score_auroc, sync_dist=True)
            self.log(f"Val/{task_name}_AP", score_ap, sync_dist=True)
            self.log(f"Val/{task_name}_Loss", tmp_loss, sync_dist=True)
            losses.append(tmp_loss)

        if self.use_loss_weighting:
            total_weight = sum(losses)
            loss_weights = [x / total_weight for x in loss_weights]

            total_loss = losses[0] * loss_weights[0] +\
                losses[1] * loss_weights[1]
        else:
            total_loss = sum(losses) / len(losses)
        experiment_dir = self._trainer.log_dir.split("lightning")[0]

        
        self.log("Val/Weighted_loss", total_loss)
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
        for layer in self.classification_layers.values():
            layer.eval()

