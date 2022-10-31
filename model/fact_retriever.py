import torch
from torch import nn
from transformers import AutoModel,  AutoConfig
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

class RetrieverModel(nn.Module):    
    def __init__(self, config):
        super().__init__()
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
    
    # Deprecated Method (why is this deprecated again?)
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #get batch wise vectors from dataloader
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        labels = batch["label"]
        labels = torch.tensor(labels).to("cuda")
        
        #prepare metadata for optimized outputs
        metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
        output_dicts = self.forward(input_ids, attention_mask, segment_ids, metadata)
        return output_dicts
    
    # Deprecated Method; again, why is this deprecated? Is it because we don't have annotations
    #  in the data for it?
    '''
    # Depricated Method

    def validation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #get batch wise vectors from dataloader
        labels = batch["label"]
        labels = torch.tensor(labels).to("cuda")
        output_dicts = self.predict(batch)
        #collecting back all the logits to get the loss
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        #log is in pytorch lightening
        self.log("loss", loss.sum(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return output_dicts
    
    # Depricated Method
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dicts = self.predict(batch)
        labels = batch["label"]
        labels = torch.tensor(labels).to("cuda")
        #collecting back all the logits to get the loss
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        #log is in pytorch lightening
        self.log("loss", loss.sum(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss.sum()}
    
    # Depricated Method
    def predict_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.predictions.extend(outputs)
    # Depricated Method
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.opt_params)
        if self.lrs_params["name"] == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        else:
            raise ValueError(f"lr_scheduler {self.lrs_params} is not supported")

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step"
                    }
                }
    '''
