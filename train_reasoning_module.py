from utils import *
import torch
from torch import nn
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoConfig, AutoTokenizer, AutoModel
from model.program_generation import ProgramGeneration
from model.span_selection import SpanSelection
import yaml
import constants

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

def model_eval(dataloader, model, device):
    #model.eval() # switch to eval model, will turn off randomness like dropout
    for step, batch in enumerate(tqdm(dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        dic = pg_model.validation() 
        #logging mechanism for loss

    for step, batch in enumerate(tqdm(dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        dic = span_model.validation()
        #logging mechanism for loss
        #outputs a dictionary.. Need to deal with it
    
    #try to use inbuilt test and predict functions
    
    f1 = f1_score(batch["labels"], dic["preds"], average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, dic["labels"], dic["labels"]

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    pg_config = yaml.load(open(constants.pg_config))
    config = pg_config["model"]["init_args"]
    pg_model = ProgramGeneration(config)
    pg_model = pg_model.to(device)

    span_config = yaml.load(open(constants.span_config))
    config = span_config["model"]["init_args"]
    span_model = SpanSelection(config)
    span_model = span_model.to(device)

    lr = args.lr
    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    #Note: make this parallel
    #Train the Program Generation Model
    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            batch_loss = pg_model.train()
            #logging mechanism for loss
            #updat optimizer, backpropagation

    #Note: make this parallel
    #Finetune the Question Classification Model
    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            batch_loss = span_model.train()
            #logging mechanism for loss
            #update optimizer, backpropagation

  def test():
    z = model_eval(dataloader, model, device)
    # logging mechanism for z
    #Getting results on Fact Retriever Module(Retriever + Question Classification):
    #evaluate/predict using test set









