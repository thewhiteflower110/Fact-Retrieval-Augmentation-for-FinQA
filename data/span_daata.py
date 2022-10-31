from torch.utils.data import Dataset
from transformers import AutoTokenizer
import constants
from models.utlis.utils import get_op_const_list
import json
import torch
from typing import Dict, List, Any, Union

class SpanSelectionDataset(Dataset):
    def __init__(self,data_file,transformer_model_name,is_training,entity_name: str = "question_type"):
        self.data_file = data_file
        self.transformer_model_name = transformer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.max_seq_length = constants.max_seq_length #### add to constant as 30 if not in training config 
        self.is_training=is_training
        self.entity_name = entity_name
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
        
        if example["qa"][self.entity_name] != "span_selection":
            return None
    
        #Getting the retrieved facts by fact retriever model
        context = ""
        for idx in example["model_input"]:
            if type(idx) == int:
                context += example["paragraphs"][idx][:-1]
                context += " "

            else:
                context += example["table_description"][idx][:-1]
                context += " "
        
        question = example["qa"]["question"]
        this_id = example["uid"]
        
        #context from FR model is concatted with question
        original_question = f"Question: {question} Context: {context.strip()}"
        if "answer" in example["qa"]:
            answer = example["qa"]["answer"]
        else:
            answer = ""
        if type(answer) != str:
            answer = str(int(answer))

        original_question_tokens = original_question.split(' ')
        example ={
            "id":this_id,
            "original_question":original_question,
            "question_tokens":original_question_tokens,
            "answer":answer
        }
        return self.concatetnating(example)

    def concatetnating(self,example):
        #encoding of original question quesion after tokenizing
        input_text_encoded = self.tokenizer.encode_plus(example["original_question"],
                                    max_length=self.max_seq_length,
                                    pad_to_max_length=True)
        input_ids = input_text_encoded["input_ids"]
        input_mask = input_text_encoded["attention_mask"]
        #encoding of original label after tokenizing
        label_encoded = self.tokenizer.encode_plus(str(example["answer"]),
                                        max_length=16,
                                        pad_to_max_length=True)
        label_ids = label_encoded["input_ids"]
        
        this_input_feature = {
            "uid": example.id,
            "tokens": example.question_tokens,
            "question": example.original_question,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "label_ids": label_ids,
            "label": str(example.answer)   
        }
        return this_input_feature

def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}
    for k in examples[0].keys():
        try:
            result_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                    batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict

  
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