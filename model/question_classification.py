import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
#from torch.utils.tensorboard import SummaryWriter


class QuestionClassification(nn.Module):    
    def __init__(self,config):
        super(QuestionClassification, self).__init__()
        self.model_config = AutoConfig.from_pretrained(config["model_name"], num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], config=self.model_config)
        #self.metric = datasets.load_metric('precision')        
        self.warmup_steps = config["lr_scheduler"]["init_args"]["num_warmup_steps"]
        self.opt_params = config["optimizer"]["init_args"]
        self.lrs_params = config["lr_scheduler"]

    def forward(self, **inputs) -> List[Dict[str, Any]]:
        return self.model(**inputs)
    '''
    # Depricated Method
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        #self.log("loss", loss, on_step=True, on_epoch=True)
        
        return loss
    # Depricated Method
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
    
    #def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
    #    return self.validation_step(batch, batch_idx)
    
    # Depricated Method
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        uids = batch["uid"]
        
        return {"preds": preds, "uids": uids}

    
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
