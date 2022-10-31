from torch.utils.data import Dataset
from transformers import AutoTokenizer
import constants
import json
import torch
from typing import Dict, List, Any, Union

class QuestionClassificationDataset(Dataset):
    def __init__(self,data_file,transformer_model_name,mode):
        self.data_file = data_file
        self.transformer_model_name = transformer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.max_seq_length = constants.max_seq_length #### add to constant as 30 if not in training config 
        self.mode=mode
        self.data_all = self.read()
        self.instances = self.get_features()
    
    def read(self):
            with open(self.data_file) as f:
                data_all = json.load(f)
            return data_all

    def get_features(self):    
        examples = []
        features_all=[]
        for example in self.data_all:
            features = self.convert_example_to_feature(example)
            features_all.append(features)
            if example:
                examples.append(example)
        return features_all

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)
    
    def convert_example_to_feature(self,example):
        feature = {}
        #encoding the input question
        input_text_encoded = self.tokenizer.encode_plus(example["qa"]["question"],
                                max_length=128,
                                pad_to_max_length=True)
        input_ids = input_text_encoded["input_ids"]
        input_mask = input_text_encoded["attention_mask"]

        feature = {
            "uid": example["uid"],
            "question": example["qa"]["question"],
            "input_ids": input_ids,
            "input_mask": input_mask,
        }

        if self.mode != "predict":
            feature["labels"] = 1 if example["qa"]["question_type"] == "arithmetic" else 0
        
        return feature
  
#helper to collate function
def right_pad_sequences(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: Union[int, bool] = 0, 
                       max_len: int = -1, device: torch.device = None) -> torch.Tensor:
    assert all([len(seq.shape) == 1 for seq in sequences])
    max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
    device = device if device is not None else sequences[0].device

    padded_seqs = []
    for seq in sequences:
        padded_seqs.append(torch.cat(seq, (torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device))))
    return torch.stack(padded_seqs)


def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}
    for k in examples[0].keys():
        try:
            if k == "labels":
                result_dict[k] = torch.tensor([example[k] for example in examples])
            else:   
                result_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                        batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict
