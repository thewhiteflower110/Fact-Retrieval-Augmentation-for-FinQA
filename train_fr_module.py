import time, random, numpy as np, argparse, sys, re, os
import torch
from torch import nn
from torch.cuda import device_count
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.fact_retriever import RetrieverModel
from model.question_classification import QuestionClassification
import yaml
import constants
from post_retriever import retriever_evaluate
from evaluation.retriever_eval import retriever_eval
from torch.utils.tensorboard import SummaryWriter
from data.retriever_data import RetrieverDataset, customized_retriever_collate_fn

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
#  Accuracy and F1 score are not the most useful metrics here. Instead we'll want Top-N
#  retrieved accuracy.
def retriever_model_eval(dataloader, retriever, device,data_file,topn):
    retriever.eval() # switch to eval model, will turn off randomness like dropout
    final_loss = 0
    criterion_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    testing_output_dict={}
    for step, batch in enumerate(tqdm(dataloader, desc=f'val', disable=TQDM_DISABLE)):            
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["input_mask"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
        output_dicts = retriever() #intuitively calling the forward method
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = criterion_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        total_loss = loss.sum()
        #Logging of Loss here
        final_loss+=total_loss
        testing_output_dict.extend(output_dicts)

    avg_final_loss = final_loss/step
    f1 = f1_score(output_dicts["labels"], output_dicts["preds"], average='macro')
    acc = accuracy_score(output_dicts["labels"], output_dicts["preds"])
    retriever_evaluate(testing_output_dict,data_file,topn)
    recall = retriever_eval(data_file)
    return acc, f1, output_dicts["labels"], output_dicts["preds"], avg_final_loss

# Is there a way to retrieve the actual operations here? That could be an important part
#  of error analysis.
def qc_model_eval(dataloader, questionClassificationModel, device):
    questionClassificationModel.eval()
    #try to use inbuilt test and predict functions
    #Getting results on Fact Retriever Module:
    val_loss = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f'val', disable=TQDM_DISABLE)):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["input_mask"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
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

writer = SummaryWriter('runs/Fact Retriever Module')

def train_log(dict):
    writer.add_scalar("Loss/train", dict["loss"])
    writer.add_scalar("Loss/Val", dict["val_loss"], dict["epoch"])

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    #### Load data
    # create the data and its corresponding datasets and dataloader

    #### Init model
    # Initiate and configure retriever
    with open(constants.retriever_config, 'r') as file:
        retriever_config = yaml.safe_load(file)    
    #config = retriever_config["model"]["init_args"]
    retriever = RetrieverModel(retriever_config["model"]["init_args"])
    retriever = retriever.to(device)

    train_dataset = RetrieverDataset(args.train,retriever_config["data"]["init_args"]["transformer_model_name"])
    dev_dataset = RetrieverDataset(args.dev, retriever_config["data"]["init_args"]["transformer_model_name"])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=retriever_config["data"]["init_args"]["batch_size"],
                                  collate_fn=customized_retriever_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=retriever_config["data"]["init_args"]["val_batch_size"],
                                collate_fn=customized_retriever_collate_fn)

    '''
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
            train_loss+=loss
            train_log({"loss":loss, "epoch":epoch})
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            #logging mechanism for loss
        #epoch wise loss logs here--
        #scheduler.step()
        
        # Double-check; are these the actual outputs of the QC model? Aren't there
        #  several different accuracy measures for that model?
        acc, f1, y_true, y_preds, avg_loss = qc_model_eval(val_dataloader, questionClassificationModel, device)
        train_log({"val_loss":avg_loss, "epoch":epoch})
    filepath="./"
    save_model(questionClassificationModel, optimizer, qc_config, filepath)
    '''
    #Note: make this parallel
    #Finetune the Retriever Model
    opt_params = retriever_config["model"]["init_args"]["optimizer"]["init_args"]
    lrs_params = retriever_config["model"]["init_args"]["lr_scheduler"]
    optimizer = configure_optimizers(retriever,opt_params,lrs_params)
    criterion_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    training_output_dict={}
    for epoch in range(args.epochs):
        #this loop discard the .train() method
        train_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):            
            input_ids = torch.tensor(batch["input_ids"]).to("cuda")
            attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
            labels = torch.tensor(batch["labels"]).to("cuda")
            segment_ids = batch["segment_ids"].to("cuda")
            metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
            output_dicts = retriever.train(input_ids, attention_mask, segment_ids, metadata) #intuitively calling the forward method
            logits = []
            for output_dict in output_dicts:
                logits.append(output_dict["logits"])
            logits = torch.stack(logits)
            loss = criterion_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss = loss.sum()
            total_loss.backward()
            train_loss+=total_loss
            train_log({"loss":total_loss, "epoch":epoch})
            # Adjust learning weights
            optimizer.step()
            training_output_dict.extend(output_dicts)
            #logging mechanism for loss
        #epoch wise loss logs here--
    res,res_message = retriever_evaluate(training_output_dict,args.train,args.topn)
    filepath="./"
    save_model(retriever, optimizer, retriever_config, filepath)
    
    #Getting val results on Fact Retriever Module(Retriever + Question Classification):
    acc, f1, y_true, y_preds,avg_loss = retriever_model_eval(dev_dataloader, retriever, device)
    train_log({"val_loss":avg_loss, "epoch":epoch})
    #log the metrics

# Why group the fact retriever and the question classification module together like this?
#  The QC model is a subsection of the Reasoning Module.
def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        retriever = RetrieverModel(config)
        retriever = retriever.to(device)
        retriever.load_state_dict(saved['retriever'])

        test_dataset = RetrieverDataset(args.dev, config["data"]["init_args"]["transformer_model_name"])

        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config["data"]["init_args"]["batch_size"],
                                  collate_fn=customized_retriever_collate_fn)
        '''
        questionClassificationModel = QuestionClassification(config)
        questionClassificationModel = questionClassificationModel.to(device)
        questionClassificationModel.load_state_dict(saved['questionClassificationModel'])
        '''

        print(f"load model from {args.filepath}")
        
        #get the dataset and dataloaders
        # Need to know what the type and origin of "test_dataloader" is? Is it a function?
        #  An iterable dict? A numpy array? Where is it's value set?
        #qc_acc, qc_f1, qc_y_true, qc_y_preds, qc_avg_loss = qc_model_eval(test_dataloader, questionClassificationModel, device)
        retriever_acc, retriever_f1, retriever_y_true, retriever_y_preds,retriever_avg_loss = retriever_model_eval(test_dataloader, retriever, device, args.test,args.topn)
        
        with open(args.test_out, "w+") as f:
            print(f"test acc :: {retriever_acc :.3f}")
            for t, p in zip( retriever_y_true, retriever_y_preds):
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
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--topn", type=int, default=3)    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    #seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
