from model.utils import utils
import torch
from torch import nn
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoConfig, AutoTokenizer, AutoModel
from model.program_generation import ProgramGeneration
from model.span_selection import SpanSelection
import yaml
import constants
import time, random, numpy as np, argparse, sys, re, os
from torch.cuda import device_count
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm


TQDM_DISABLE=True
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pg_model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    total_loss =0 
    for step, batch in enumerate(tqdm(dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        program_ids = batch["program_ids"]
        program_mask = batch["program_mask"]
        option_mask = batch["option_mask"]
        is_training = False
        output_dicts = model(is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, metadata)
        
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = criterion(logits.view(-1, logits.shape[-1]), program_ids.view(-1))
        loss = loss * program_mask.view(-1)
        total_loss +=loss
        #logging mechanism for loss
    #f1 = f1_score(batch["labels"], logits, average='macro')
    #acc = accuracy_score(batch["labels"], dic["preds"])
    avg_total_loss = total_loss/step
    return avg_total_loss

def span_selection_model_eval(dataloader, model, device):
    total_loss =0 
    for step, batch in enumerate(tqdm(dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        label_ids = torch.tensor(batch["label_ids"]).to("cuda")
        loss = model(input_ids,attention_mask,label_ids) 
        #logging mechanism for loss
        #outputs a dictionary.. Need to deal with it
    
    #try to use inbuilt test and predict functions
    
    #f1 = f1_score(batch["labels"], dic["preds"], average='macro')
    #acc = accuracy_score(batch["labels"], dic["preds"])
    avg_total_loss = total_loss/step
    return avg_total_loss

def save_model(model, optimizer, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

writer = SummaryWriter('runs/Reasoning Module')

def train_log(dict):
    writer.add_scalar("Loss/train", dict["loss"])
    writer.add_scalar("Loss/Val", dict["val_loss"], epoch)

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    with open(constants.pg_config, 'r') as file:
        pg_config = yaml.safe_load(file)  
    config = pg_config["model"]["init_args"]
    pg_model = ProgramGeneration(config)
    pg_model = pg_model.to(device)
    
    with open(constants.span_config, 'r') as file:
        span_config = yaml.safe_load(file)  
    config = span_config["model"]["init_args"]
    span_model = SpanSelection(config)
    span_model = span_model.to(device)

    #Note: make this parallel
    #Train the Program Generation Model
    opt_params = pg_config["model"]["init_args"]["optimizer"]["init_args"]
    lrs_params = pg_config["model"]["init_args"]["lr_scheduler"]
    optimizer = configure_optimizers(pg_model,opt_params,lrs_params)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            input_ids = batch["input_ids"]
            input_mask = batch["input_mask"]
            segment_ids = batch["segment_ids"]
            program_ids = batch["program_ids"]
            program_mask = batch["program_mask"]
            option_mask = batch["option_mask"]
            is_training = True
            
            program_ids = torch.tensor(program_ids).to("cuda")
            program_mask = torch.tensor(program_mask).to("cuda")
            
            metadata = [{"unique_id": filename_id} for filename_id in batch["unique_id"]]
            
            output_dicts = pg_model(is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, metadata)
            
            logits = []
            for output_dict in output_dicts:
                logits.append(output_dict["logits"])
            logits = torch.stack(logits)
            loss = criterion(logits.view(-1, logits.shape[-1]), program_ids.view(-1))
            loss = loss * program_mask.view(-1)
            #logging mechanism for loss
            train_log({"loss":loss, "epoch":epoch})
            train_loss += loss
            loss.backward()
            optimizer.step()
        loss = pg_model_eval(val_dataloader,pg_model,device)
        train_log({"val_loss":loss, "epoch":epoch})
        # logging mechanism for loss
      #epoch logs here --
    filepath="./"
    save_model(pg_model, optimizer, pg_config, filepath)

    #Note: make this parallel
    #Finetune the Question Classification Model
    opt_params = span_config["model"]["init_args"]["optimizer"]["init_args"]
    lrs_params = span_config["model"]["init_args"]["lr_scheduler"]
    optimizer = configure_optimizers(span_model,opt_params,lrs_params)
    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            input_ids = torch.tensor(batch["input_ids"]).to("cuda")
            attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
            label_ids = torch.tensor(batch["label_ids"]).to("cuda")
            loss = span_model(input_ids, attention_mask, label_ids)
            #logging mechanism for loss
            #update optimizer, backpropagation
            loss.backward()
            optimizer.step()
        loss = span_selection_model_eval(val_dataloader,span_model,device)
        #logging mechanism for loss
    filepath="./"
    save_model(span_model, optimizer, pg_config, filepath)

def test():
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        span_model = SpanSelection(config)
        span_model = span_model.to(device)
        loss = span_selection_model_eval(test_dataloader,span_model,device)
        # logging mechanism for loss
        
        saved = torch.load(args.filepath)
        config = saved['model_config']
        pg_model = ProgramGeneration(config)
        pg_model = pg_model.to(device)
        loss = pg_model_eval(test_dataloader,pg_model,device)
        # logging mechanism for loss

def configure_optimizers(model,opt_params,lrs_params):
    optimizer = AdamW(model.parameters(), **opt_params)
    if lrs_params["name"] == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **lrs_params["init_args"])
    elif lrs_params["name"] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, **lrs_params["init_args"])
    elif lrs_params["name"] == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, **lrs_params["init_args"])
    else:
        raise ValueError(f"lr_scheduler {lrs_params} is not supported")

    return {"optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
                }
            }

#change the args according to need
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
