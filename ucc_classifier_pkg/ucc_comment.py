from transformers import RobertaTokenizer
from torch.utils.data import Dataset
import pandas as pd


class UCC_Comments(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: RobertaTokenizer, max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        comment = row.comment
        encoding = self.tokenizer.encode_plus(comment, add_special_tokens=True,
                                              return_tensors='pt', truncation=True,
                                              padding='max_length', max_length=self.max_token_len,
                                              return_attention_mask=True)
        return {'comment_text': comment, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}
