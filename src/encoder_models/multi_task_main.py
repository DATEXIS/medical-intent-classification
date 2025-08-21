
from lightning.pytorch.cli import LightningCLI
#from bert_model import BertClassificationModel, MultiTaskBertClassificationModel
from src.encoder_models.datamodule import MultiTaskDataModule
from multi_task_bert import MultiTaskBertClassificationModel


cli = LightningCLI(MultiTaskBertClassificationModel, MultiTaskDataModule)