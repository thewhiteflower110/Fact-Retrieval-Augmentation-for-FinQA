import time, random, numpy as np, argparse, sys, re, os
#from types import SimpleNamespace
import torch
from torch import nn
from torch.cuda import device_count
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
#from bert import BertModel
#from optimizer import AdamW
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.fact_retriever import RetrieverModel
from model.question_classification import QuestionClassification
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

# perform model evaluation in terms of the accuracy and f1 score.
def retriever_model_eval(dataloader, retriever, device):
    retriever.eval() # switch to eval model, will turn off randomness like dropout
    final_loss = 0
    criterion_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    for step, batch in enumerate(tqdm(dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):            
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")
        output_dicts = retriever() #intuitively calling the forward method
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = criterion_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        total_loss = loss.sum()
        #Logging of Loss here
        final_loss+=total_loss

    avg_final_loss = final_loss/step
    f1 = f1_score(output_dicts["labels"], output_dicts["preds"], average='macro')
    acc = accuracy_score(output_dicts["labels"], output_dicts["preds"])

    return acc, f1, output_dicts["labels"], output_dicts["preds"], avg_final_loss

def qc_model_eval(dataloader, questionClassificationModel, device):
    questionClassificationModel.eval()
    #try to use inbuilt test and predict functions
    #Getting results on Fact Retriever Module:
    val_loss = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")
        outputs = questionClassificationModel(input_ids,attention_mask,labels) #intuitively calling the forward method
        loss = outputs.loss
        total_loss+=loss
        #Logging of Loss here

    avg_total_loss = total_loss/step
    f1 = f1_score(labels, outputs["preds"], average='macro')
    acc = accuracy_score(labels, outputs["preds"])

    return acc, f1, labels, outputs["preds"], avg_total_loss

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def model_save():
    pass

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    #### Load data
    # create the data and its corresponding datasets and dataloader
    #train_data, num_labels = create_data(args.train, 'train')
    #dev_data = create_data(args.dev, 'valid')

    #train_dataset = BertDataset(train_data, args)
    #dev_dataset = BertDataset(dev_data, args)

    #train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
    #                              collate_fn=train_dataset.collate_fn)
    #dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
    #                            collate_fn=dev_dataset.collate_fn)

    #### Init model

    with open(constants.retriever_config, 'r') as file:
        retriever_config = yaml.safe_load(file)    
    config = retriever_config["model"]["init_args"]
    retriever = RetrieverModel(config)
    retriever = retriever.to(device)

    #question classification config
    with open(constants.qc_config, 'r') as file:
        qc_config = yaml.safe_load(file)  
    config = qc_config["model"]["init_args"]
    questionClassificationModel = QuestionClassification(config)
    questionClassificationModel = questionClassificationModel.to(device)

    #Note: make this parallel
    #Finetune the Question Classification Model
    opt_params = qc_config["model"]["init_args"]["optimizer"]["init_args"]
    lrs_params = qc_config["model"]["init_args"]["lr_scheduler"]
    optimizer = configure_optimizers(questionClassificationModel,opt_params,lrs_params)
    
    for epoch in range(args.epochs):
        #this loop discard the .train() method
        train_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):            
            input_ids = torch.tensor(batch["input_ids"]).to("cuda")
            attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
            labels = torch.tensor(batch["labels"]).to("cuda")
            outputs = questionClassificationModel.train() #intuitively calling the forward method
            loss = outputs.loss
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            #logging mechanism for loss
        #epoch wise loss logs here--
        #scheduler.step()
        acc, f1, y_true, y_preds, avg_loss = qc_model_eval(val_dataloader, questionClassificationModel, device)

    PATH ="./"
    torch.save(questionClassificationModel.state_dict(), PATH)
    #Note: make this parallel
    #Finetune the Question Classification Model
    for epoch in range(args.epochs):
        #this loop discard the .train() method
        train_loss = 0
        num_batches = 0
        opt_params = retriever_config["model"]["init_args"]["optimizer"]["init_args"]
        lrs_params = retriever_config["model"]["init_args"]["lr_scheduler"]
        optimizer = configure_optimizers(retriever,opt_params,lrs_params)
        criterion_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):            
            input_ids = torch.tensor(batch["input_ids"]).to("cuda")
            attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
            labels = torch.tensor(batch["labels"]).to("cuda")
            output_dicts = retriever.train() #intuitively calling the forward method
            logits = []
            for output_dict in output_dicts:
                logits.append(output_dict["logits"])
            logits = torch.stack(logits)
            loss = criterion_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss = loss.sum()
            total_loss.backward()
            # Adjust learning weights
            optimizer.step()
            #logging mechanism for loss
        #epoch wise loss logs here--
    
    PATH ="./"
    torch.save(retriever.state_dict(), PATH)
    
    #Getting val results on Fact Retriever Module(Retriever + Question Classification):
    acc, f1, y_true, y_preds,avg_loss = retriever_model_eval(val_dataloader, retriever, device)
    #log the metrics

def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        retriever = RetrieverModel(config)
        retriever = retriever.to(device)
        questionClassificationModel = QuestionClassification(config)
        questionClassificationModel = questionClassificationModel.to(device)

        retriever.load_state_dict(saved['questionClassificationModel'])
        retriever.load_state_dict(saved['retriever'])
        questionClassificationModel = questionClassificationModel.to(device)
        retriever = retriever.to(device)
        print(f"load model from {args.filepath}")
        
        #get the dataset and dataloaders
        qc_acc, qc_f1, qc_y_true, qc_y_preds, qc_avg_loss = qc_model_eval(test_dataloader, questionClassificationModel, device)
        retriever_acc, retriever_f1, retriever_y_true, retriever_y_preds,retriever_avg_loss = retriever_model_eval(test_dataloader, retriever, device)
        
        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for t, p in zip( test_true, test_pred):
                f.write(f"{t} ||| {p}\n")

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
