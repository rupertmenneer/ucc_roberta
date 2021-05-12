import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import RobertaModel


class UCC_CommentClassifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        # RoBERTa Layer
        self.bert = RobertaModel.from_pretrained(config['model_name'], return_dict=True)
        # Final classifier layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, config['n_classes'])
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # roberta layer
        output = self.bert(input_ids, attention_mask=attention_mask)
        # mean of hidden states
        pooled_output = torch.mean(output.last_hidden_state, 1)
        # pass to classifier
        output = self.classifier(pooled_output)
        return output

    def training_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        pass

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        print(batch_idx)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        predictions = self(input_ids, attention_mask)
        predictions = torch.sigmoid(predictions)
        return predictions
