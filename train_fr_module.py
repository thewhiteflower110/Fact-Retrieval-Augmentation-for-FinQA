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
from post_retriever import retriever_evaluate, combine_all_outputs
from evaluation.retriever_eval import retriever_eval
from torch.utils.tensorboard import SummaryWriter
from data.retriever_data import RetrieverDataset, customized_retriever_collate_fn
from torchmetrics import F1Score

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
    testing_output_dict=[]
    all_labels = []
    all_logits = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'val', disable=TQDM_DISABLE)):            
        labels = torch.tensor(batch["label"]).to(device)
        metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
        output_dicts = retriever(batch["input_ids"], batch["input_mask"], batch["segment_ids"], metadata, device) #intuitively calling the forward method
        logits = []
        logits_=[]
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
            logits_.append(output_dict["logits"].argmax())
        logits = torch.stack(logits)
        loss = criterion_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        total_loss = loss.sum() #total loss of entire batch
        final_loss+=total_loss #total loss of validation
        testing_output_dict.extend(output_dicts)
        all_labels.extend(batch["label"])
        all_logits.extend(logits_)

    avg_final_loss = final_loss/step
    #final_dict = combine_all_outputs(testing_output_dict)
    f1 = F1Score(num_classes=2)
    #f1(, all_labels)
    f1 = f1_score(all_labels, all_logits, average='macro')
    acc = accuracy_score(all_labels, all_logits)
    retriever_evaluate(testing_output_dict,data_file,topn)
    #recall = retriever_eval(data_file)
    return acc, f1,all_labels, all_logits, avg_final_loss

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

def save_model(name, model, optimizer, args, config, filepath):
    save_info = {
        'name': name,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    files=filepath+"/"+name+".pt"
    torch.save(save_info, files)
    print(f"save the model to {filepath}")

writer = SummaryWriter('runs/Fact Retriever Module')

def train_log(dict):
    if "val_loss" in dict:
      writer.add_scalar("Loss/Val", dict["val_loss"], dict["epoch"])
    else:
      writer.add_scalar("Loss/train", dict["loss"], dict["epoch"])
    
def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    #### Load data
    # create the data and its corresponding datasets and dataloader

    #### Init model
    # Initiate and configure retriever
    with open(constants.retriever_config, 'r') as file:
        retriever_config = yaml.safe_load(file)    
    #config = retriever_config["model"]["init_args"]
    retriever = RetrieverModel(retriever_config)
    retriever = retriever.to(device)

    train_dataset = RetrieverDataset(args.train,retriever_config["data"]["init_args"]["transformer_model_name"],mode="train")
    dev_dataset = RetrieverDataset(args.dev, retriever_config["data"]["init_args"]["transformer_model_name"],mode="valid")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=retriever_config["data"]["init_args"]["batch_size"],
                                  collate_fn=customized_retriever_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=retriever_config["data"]["init_args"]["val_batch_size"],
                                collate_fn=customized_retriever_collate_fn)

    
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
    
    #Note: make this parallel
    #Finetune the Retriever Model
    opt_params = retriever_config["model"]["init_args"]["optimizer"]["init_args"]
    lrs_params = retriever_config["model"]["init_args"]["lr_scheduler"]
    scheduler = configure_optimizers(retriever,opt_params,lrs_params)
    criterion_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    training_output_dict=[]
    for epoch in range(args.epochs):
        #this loop discard the .train() method
        train_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):            
            #input_ids = torch.tensor(batch["input_ids"]).to(device)
            #attention_mask = torch.tensor(batch["input_mask"]).to(device)
            labels = torch.tensor(batch["label"]).to(device)
            #segment_ids = torch.tensor(batch["segment_ids"]).to(device)
            metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
            output_dicts = retriever.forward(batch["input_ids"], batch["input_mask"], batch["segment_ids"], metadata, device) #intuitively calling the forward method
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
            scheduler.step()
            training_output_dict.extend(output_dicts)
            #logging mechanism for loss
        #epoch wise loss logs here--
        #Getting val results on Fact Retriever Module(Retriever + Question Classification):
        acc, f1, y_true, y_preds,avg_loss = retriever_model_eval(dev_dataloader, retriever, device,args.dev,args.topn)
        print(acc,f1,avg_loss)
        train_log({"val_loss":avg_loss, "epoch":epoch})
        #log the metrics

    retriever_evaluate(training_output_dict,args.train,args.topn)
    if not os.path.exists(args.save_model_path):
        # Create a new directory because it does not exist
        os.makedirs(args.save_model_path)
    save_model("FactRetriever", retriever, scheduler, args, retriever_config, args.save_model_path)

    
# Why group the fact retriever and the question classification module together like this?
#  The QC model is a subsection of the Reasoning Module.
def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        name="FactRetriever"
        path= args.save_model_path+"/"+name+".pt"
        #path= args.save_model_path+"/name.pt"
        saved = torch.load(path)
        config = saved['model_config']
        retriever = RetrieverModel(config)
        retriever = retriever.to(device)
        retriever.load_state_dict(saved["model"])

        test_dataset = RetrieverDataset(args.test, config["data"]["init_args"]["transformer_model_name"])

        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config["data"]["init_args"]["batch_size"],
                                  collate_fn=customized_retriever_collate_fn)
        
        questionClassificationModel = QuestionClassification(config)
        questionClassificationModel = questionClassificationModel.to(device)
        questionClassificationModel.load_state_dict(saved['questionClassificationModel'])
        

        print(f"load model from {path}")
        
        #get the dataset and dataloaders
        # Need to know what the type and origin of "test_dataloader" is? Is it a function?
        #  An iterable dict? A numpy array? Where is it's value set?
        #qc_acc, qc_f1, qc_y_true, qc_y_preds, qc_avg_loss = qc_model_eval(test_dataloader, questionClassificationModel, device)
        retriever_acc, retriever_f1, retriever_y_true, retriever_y_preds,retriever_avg_loss = retriever_model_eval(test_dataloader, retriever, device, args.test,args.topn)
        print("Test values here")
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
    return lr_scheduler

#change the args according to need
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--topn", type=int, default=3)    
    parser.add_argument("--epochs", type=int, default=1)   
    parser.add_argument("--save_model_path", type=str, default="./checkpoints")  
    parser.add_argument("--test_out", type=str, default="results.txt")  

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    #seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
