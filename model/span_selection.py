import torch
from typing import Optional, Dict, Any, Tuple, List
from transformers import T5ForConditionalGeneration, AutoTokenizer
#from torch.utils.tensorboard import SummaryWriter


class SpanSelection:    
    def __init__(self,given_model,optimizer,lr_scheduler,load_ckpt_file,test_set,input_dir ):
        self.tokenizer = AutoTokenizer.from_pretrained(given_model)
        self.model = T5ForConditionalGeneration.from_pretrained(given_model)
        self.lrs_params = lr_scheduler
        self.opt_params = optimizer["init_args"]
        
        self.test_set = test_set
        self.input_dir = input_dir
        #self.writer = SummaryWriter()
        self.predictions = []

    def forward(self, input_ids, attention_mask, label_ids) -> List[Dict[str, Any]]:
        input_ids = torch.tensor(input_ids).to("cuda")
        attention_mask = torch.tensor(attention_mask).to("cuda")
        label_ids = torch.tensor(label_ids).to("cuda")
        
        loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels = label_ids).get("loss")
        return {"loss": loss}

    def train(self, batch: torch.Tensor)-> List[Dict[str, Any]]:
        #def train(self, input_ids, attention_mask, label_ids) -> List[Dict[str, Any]]:
        #input_ids = torch.tensor(input_ids).to("cuda")
        #attention_mask = torch.tensor(attention_mask).to("cuda")
        #label_ids = torch.tensor(label_ids).to("cuda")
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        label_ids = torch.tensor(batch["label_ids"]).to("cuda")

        loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels = label_ids).get("loss")
        #self.writer.add_scalar("Loss/train", loss)
        #self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation(self, batch: torch.Tensor):
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        label_ids = torch.tensor(batch["label_ids"]).to("cuda")
        labels = batch["label"]

        #Generating ids using model
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        #tokenizing the generated ids.
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        
        output_dict = self.forward(input_ids, attention_mask, label_ids = label_ids)
        
        unique_ids = batch["uid"]
        output_dict["preds"] = {}
        for i, unique_id in enumerate(unique_ids):
            output_dict["preds"][unique_id] = (preds[i], labels[i])
            
        loss = output_dict["loss"]
        #self.writer.add_scalar("Loss/valid", loss)
        #self.log("val_loss", loss)
        return output_dict
    
    def predict(self, batch: torch.Tensor):
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        
        unique_ids = batch["uid"]
        output_dict = []
        for i, unique_id in enumerate(unique_ids):
            output_dict.append({"uid": unique_id, "preds": preds[i]})
        return output_dict

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
