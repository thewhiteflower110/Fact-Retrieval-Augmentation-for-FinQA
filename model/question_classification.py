import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
#from torch.utils.tensorboard import SummaryWriter


class QuestionClassification:    
    def __init__(self,given_model,warmup_steps,optimizer,lr_scheduler):
        '''
        Metrics in the paper--->
        topn: 10
        dropout_rate: 0.1
        optimizer:
        init_args: 
            lr: 2.0e-5
            betas: 
            - 0.9
            - 0.999
            eps: 1.0e-8
            weight_decay: 0.1
        lr_scheduler:
        name: linear
        init_args:
            num_warmup_steps: 100
            num_training_steps: 10000

        '''
        self.model_config = AutoConfig.from_pretrained(given_model, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(given_model, config=self.model_config)
        #self.metric = datasets.load_metric('precision')        
        self.warmup_steps = warmup_steps
        self.opt_params = optimizer["init_args"]
        self.lrs_params = lr_scheduler
    
    def forward(self, **inputs) -> List[Dict[str, Any]]:
        return self.model(**inputs)

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        #self.log("loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        #log is in pytorch lightening
        #self.log("loss", loss, on_step=True, on_epoch=True)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        labels = batch["labels"]
        uids = batch["uid"]
        
        return {"preds": preds, "labels": labels, "uids": uids}
    '''
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)
    '''

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        uids = batch["uid"]
        
        return {"preds": preds, "uids": uids}

    '''
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
