from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from typing import Dict, Iterable, List, Any, Optional, Union
import random
from tqdm import tqdm
import retriever_helper
import torch

class RetrieverDataset(Dataset):
    def __init__(self, 
        transformer_model_name: str,
        file_path: str,
        max_instances: int,
        mode: str = "train", 
        **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "test", "valid"]

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        self.max_instances = max_instances
        self.mode = mode
        self.instances = self.read(file_path, self.tokenizer)

    def read(self, input_path: str, tokenizer) -> Iterable[Dict[str, Any]]:
        with open(input_path) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)

        examples = []
        for entry in input_data:
            examples.append(retriever_helper.read_mathqa_entry(entry, tokenizer))

        if self.mode == "train":
            kwargs = {"examples": examples,
            "tokenizer": tokenizer,
            "option": "rand",
            "is_training": True,
            "max_seq_length": 512,
            }
        else:
            kwargs = {"examples": examples,
            "tokenizer": tokenizer,
            "option": "rand",
            "is_training": False,
            "max_seq_length": 512,
            }

        features = self.convert_examples_to_features(**kwargs)
        data_pos, neg_sent, irrelevant_neg_table, relevant_neg_table = features[0], features[1], features[2], features[3]

        
        if self.mode == "train":
            random.shuffle(neg_sent)
            random.shuffle(irrelevant_neg_table)
            random.shuffle(relevant_neg_table)
            data = data_pos + relevant_neg_table[:min(len(relevant_neg_table),len(data_pos) * 3)] + irrelevant_neg_table[:min(len(irrelevant_neg_table),len(data_pos) * 2)] + neg_sent[:min(len(neg_sent),len(data_pos))]
        else:
            data = data_pos + neg_sent + irrelevant_neg_table + relevant_neg_table
        print(self.mode, len(data))
        return data

    def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 option,
                                 is_training,
                                 ):
        """Converts a list of DropExamples into InputFeatures."""
        res, res_neg_sent, res_irrelevant_neg_table, res_relevant_neg_table = [], [], [], []
        for (example_index, example) in tqdm(enumerate(examples)):
            pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = example.convert_single_example(
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                option=option,
                is_training=is_training,
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token)

            res.extend(pos_features)
            res_neg_sent.extend(neg_sent_features)
            res_irrelevant_neg_table.extend(irrelevant_neg_table_features)
            res_relevant_neg_table.extend(relevant_neg_table_features)
            
        return res, res_neg_sent, res_irrelevant_neg_table, res_relevant_neg_table


    def __len__(self):
        return self.instances.shape[1]

    def __getitem__(self, idx):
        return self.instances[idx]
        
def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}
    for k in examples[0].keys():
        try:
            result_dict[k] = retriever_helper.right_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                    batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict



