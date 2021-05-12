from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ucc_classifier_pkg.ucc_comment import UCC_Comments


class UCC_CommentDataModule(pl.LightningDataModule):

    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(self, predict_data, tokenizer, batch_size=16, max_token_len=128):
        super().__init__()
        self.predict_data = predict_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    # This class creates a dataset for our incoming data, making sure to pass correct encodings back to our model

    def setup(self):
        self.predict_dataset = UCC_Comments(
            self.predict_data,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        pass
    def val_dataloader(self):
        pass
    def test_dataloader(self):
        pass
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=4)