'''
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import json

def read(data_file):
  with open(data_file) as f:
      data_all = json.load(f)
  return data_all

data_file="/content/dev2.json"
data_all = read(data_file)
text_all = []
for i in data_all:
  question = i["qa"]["question"] 
  paragraphs = i["paragraphs"] #list of sentences here
  if 'text_evidence' in i["qa"]:
      pos_sent_ids = i["qa"]['text_evidence']

  context = question+paragraphs
  text_all.append(context)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
inputs = tokenizer(text_all, padding=True,truncation = True, return_tensors='pt')
outputs = model(**inputs,output_hidden_states=True )
last_hidden_state = outputs.hidden_states[-1]
cls_token = last_hidden_state[0,0,:]
print(cls_token.shape)
'''
import torch
from torch import nn
from transformers import AutoModel,  AutoConfig
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

class RetrieverModel(nn.Module):    
    def __init__(self, config):
        super().__init__()
        config=config["model"]["init_args"]
        self.model = AutoModel.from_pretrained(config["transformer_model_name"])
        self.model_config = AutoConfig.from_pretrained(config["transformer_model_name"])
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        hidden_size = self.model_config.hidden_size
        
        self.topn = config["topn"]
        self.dropout_rate = config["dropout_rate"]
        self.warmup_steps = config["lr_scheduler"]["init_args"]["num_warmup_steps"]
        self.opt_params = config["optimizer"]["init_args"]
        self.lrs_params = config["lr_scheduler"]
        #self.predictions = []

        # Classifies whether a specific fact is relevant supporting info for the question
        #  Facts that we want to retrieve should have highly positive logits,
        #  facts that we want to ignore should have highly negative logits, ideally.
        self.classifier = nn.Sequential(
          nn.Linear(hidden_size, hidden_size, bias=True),
          nn.Dropout(self.dropout_rate),
          nn.Linear(hidden_size, 2, bias=True)
        )
    
    def forward(self, input_ids, attention_mask, segment_ids, metadata, device) -> List[Dict[str, Any]]:
        
        #put the 3 vectors on gpu
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        segment_ids = torch.tensor(segment_ids).to(device)

        # Get encodings from the Language Model
        bert_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        
        #get the cls token encoding
        bert_sequence_output = bert_outputs.last_hidden_state
        #get pooled output off the cls encoding
        bert_pooled_output = bert_sequence_output[:, 0, :]
        # pass through trainable classifier
        logits = self.classifier(bert_pooled_output)

        #give individual outputs according to file. 
        # 1 means the file has important facts for the question
        output_dicts = []
        # Can we get a format for the "metadata" written down somewhere? 
        for i in range(len(metadata)):
            output_dicts.append({"logits": logits[i], "filename_id": metadata[i]["filename_id"], "ind": metadata[i]["ind"]})
        return output_dicts
