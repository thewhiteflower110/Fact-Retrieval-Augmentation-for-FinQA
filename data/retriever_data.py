# RETRIEVER Dataset and dataloader!!
import json
from transformers import AutoTokenizer
import constants
import re
import torch
from typing import Dict, Iterable, List, Any, Optional, Union
from torch.utils.data import Dataset

class RetrieverDataset(Dataset):
    def __init__(self,data_file,transformer_model_name):
        self.data_file = data_file
        self.data_all = self.read()
        self.transformer_model_name = transformer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.max_seq_length = constants.max_seq_length #### add to constant as 30
        self.instances = self.get_features()

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances) 
    def read(self):
        with open(self.data_file) as f:
            data_all = json.load(f)
        return data_all
    
    def get_features(self):    
        #features_all=[]
        res, res_neg_sent, res_irrelevant_neg_table, res_relevant_neg_table = [], [], [], []
        for example in self.data_all:
            features = self.get_example_feature(example)
            pos_sent_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = features[0], features[1], features[2], features[3]
            # MODEL WRITE extend, we write APPEND
            res.extend(pos_sent_features) 
            res_neg_sent.extend(neg_sent_features)
            res_irrelevant_neg_table.extend(irrelevant_neg_table_features)
            res_relevant_neg_table.extend(relevant_neg_table_features)
        
        return res, res_neg_sent, res_irrelevant_neg_table, res_relevant_neg_table

    def get_example_feature(self,example):
      question = example["qa"]["question"]
    
      paragraphs = example["paragraphs"]
      # tables = entry["tables"]
      
      if 'text_evidence' in example["qa"]:
          pos_sent_ids = example["qa"]['text_evidence']
          pos_table_ids = example["qa"]['table_evidence']
      else: # test set
          pos_sent_ids = []
          pos_table_ids = []

      
      table_descriptions = example["table_description"]
      filename_id = example["uid"]

      example= {
          "filename_id":filename_id,
          "question":question,
          "paragraphs":paragraphs,
          # tables=tables,
          "table_descriptions":table_descriptions,
          "pos_sent_ids":pos_sent_ids,
          "pos_table_ids":pos_table_ids}

      return self.convert_example_to_feature(example)

    #tokenizes the given text and takes care of special tokens
    def tokenize(self,tokenizer, text, apply_basic_tokenization=False):
        """Tokenizes text, optionally looking up special tokens separately.
        Args:
        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
        text: text to tokenize
        apply_basic_tokenization: If True, apply the basic tokenization. If False,
            apply the full tokenization (basic + wordpiece).
        Returns:
        tokenized text.
        A special token is any text with no spaces enclosed in square brackets with no
        space, so we separate those out and look them up in the dictionary before
        doing actual tokenization.
        """

        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

        #bert basic tokenizer 
        tokenize_fn = tokenizer.tokenize
        if apply_basic_tokenization:
            tokenize_fn = tokenizer.basic_tokenizer.tokenize

        #tokenize the usual text, and for special tokens, use them if present or use UNK
        tokens = []
        for token in text.split(" "):
            if _SPECIAL_TOKENS_RE.match(token):
                #put the special tokens as it
                if token in tokenizer.get_vocab():
                    tokens.append(token)
                # put the special tokens as UNKs
                else:
                    tokens.append(tokenizer.unk_token)
            else:
                tokens.extend(tokenize_fn(token))
        return tokens

    #concatenating each sentence with its question and converting to model facourable inputs
    def concatenating(self,tokenizer, question, sent, label, max_seq_length,cls_token, sep_token):
        '''
        single pair of question, context, label feature
        ##  DONT KNOW WHAT IS cls token and sep token!!
        '''

        question_tokens = self.tokenize(tokenizer, question)
        context_tokens = self.tokenize(tokenizer, sent)
        tokens = [cls_token] + question_tokens + [sep_token] + context_tokens
        #initialized to 0
        segment_ids = [0] * len(tokens)
        #segment_ids.extend([0] * len(context_tokens))
        
        #striping the input till max seq length
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length-1] #29
            tokens += [sep_token] #30th token
            segment_ids = segment_ids[:max_seq_length]
        
        #getting ids of token and initializing the mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        #converting to max_seq_lengtj
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)
        
        this_input_feature = {
            "context": sent, #list of sentences
            "tokens": tokens, #list of words
            "input_ids": input_ids,#list of ints
            "input_mask": input_mask,#list
            "segment_ids": segment_ids, #list
            "label": label
        }
        return this_input_feature

    def convert_example_to_feature(self,example):
        #1 example
        #returns 4 list of tensors/lists ?
        pos_sent_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = [], [], [], []
        question = example["question"]
        paragraphs = example["paragraphs"]
        pos_text_ids = example["pos_sent_ids"]
        pos_table_ids = example["pos_table_ids"] #true table evidence indexes
        table_descriptions = example["table_descriptions"]
        relevant_table_ids = set([i.split("-")[0] for i in pos_table_ids])
        cls_token =self.tokenizer(cls_token) ##CHANGE THIS IF IT DOESNT WORK !
        sep_token = self.tokenizer(sep_token) ### CHANGE THIS
        #paragraphs
        #COnverting each sentence to input feature with thier sentence ids
        for sent_idx, sent in enumerate(paragraphs):
            if sent_idx in pos_text_ids:

                this_input_feature = self.concatenating(
                    self.tokenizer, question, sent, 1, self.max_seq_length,
                    cls_token, sep_token)
            else:
                this_input_feature = self.concatenating(
                    self.tokenizer, question, sent, 0, self.max_seq_length,
                    cls_token, sep_token)
            
            this_input_feature["ind"] = sent_idx
            this_input_feature["filename_id"] = example["filename_id"]
            
            if sent_idx in pos_text_ids:
                pos_sent_features.append(this_input_feature)
            else:
                neg_sent_features.append(this_input_feature)

        #tables
        
        for cell_idx in table_descriptions:
            this_gold_sent = table_descriptions[cell_idx]
            if cell_idx in pos_table_ids:
                this_input_feature = self.concatenating(
                    self.tokenizer, question, this_gold_sent, 1, self.max_seq_length,
                    cls_token, sep_token)
                this_input_feature["ind"] = cell_idx
                this_input_feature["filename_id"] = example.filename_id
                pos_sent_features.append(this_input_feature)
            else:
                ti = cell_idx.split("-")[0]
                this_input_feature = self.concatenating(
                    self.tokenizer, question, this_gold_sent, 0, self.max_seq_length,
                    cls_token, sep_token)
                this_input_feature["ind"] = cell_idx
                this_input_feature["filename_id"] = example.filename_id
                # even if exact cell_idx is not present it is trying to find if prefix 0 exists in table indx, and it will keep it as relevant
                if ti in relevant_table_ids:
                    relevant_neg_table_features.append(this_input_feature)
                else:
                    irrelevant_neg_table_features.append(this_input_feature)
        
        return pos_sent_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features

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

def customized_retriever_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}
    for k in examples[0].keys():
        try:
            result_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                    batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict
