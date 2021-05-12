import pytorch_lightning as pl
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np
from typing import List

from ucc_classifier_pkg.ucc_classifier import UCC_CommentClassifier
from ucc_classifier_pkg.ucc_datamodule import UCC_CommentDataModule


class ClassifierWrapper:

    default_config = {
        'model_name': 'roberta-base',
        'attributes': ['antagonise', 'condescending', 'dismissive', 'generalisation',
                       'generalisation_unfair', 'unhealthy', 'hostile', 'sarcastic'],
        'n_classes': 8,
        'n_gpus': 0,
        'batch_size': 32,
        'max_token_len': 128,
        'checkpoint_dir': "/Users/rupert/Documents/ucc_classifier/assets/roberta-ucc-model-v1.0.0.ckpt"
    }

    def __init__(self, config=default_config):
        self.config = config
        print("Tokenizer loading...")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config['model_name'])
        print("Tokenizer loaded successfully")
        # load model from checkpoint
        print("Model loading...")
        self.model = UCC_CommentClassifier.load_from_checkpoint(self.config['checkpoint_dir'], config=self.config)
        # we freeze the model weights/biases here because we have already trained
        # the model and only using the model for inference
        print("Model loaded successfully")
        self.model.freeze()
        self.trainer = pl.Trainer(gpus=self.config['n_gpus'], num_sanity_val_steps=0, progress_bar_refresh_rate=30)

    # method to convert list of string comments to pytorch data module
    def _create_data_module(self, comments):
        comments = pd.DataFrame(data=comments, columns=['comment'])
        data_module = UCC_CommentDataModule(predict_data=comments,
                                                  tokenizer=self.tokenizer,
                                                  batch_size=self.config['batch_size'],
                                                  max_token_len=self.config['max_token_len'])
        data_module.setup()
        return data_module

    # method to convert list of comments into predictions for each comment
    def classify_raw_comments(self, comments: List[str], return_type='bool'):
        data_module = self._create_data_module(comments)
        print("Datamodule created successfully")
        print("Starting to predict...")
        predictions = self.trainer.predict(self.model, datamodule=data_module)
        print("Predictions finished successfully...")
        if return_type == 'bool':
            flattened_predictions = [p > 0.5 for batch in predictions for p in batch]
        elif return_type == 'float':
            flattened_predictions = [p for batch in predictions for p in batch]
        flattened_predictions = np.stack(flattened_predictions)
        return flattened_predictions
