import time, random, numpy as np, argparse, sys, re, os
#from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
#from bert import BertModel
from optimizer import AdamW
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
def model_eval(dataloader, model, device):
    #model.eval() # switch to eval model, will turn off randomness like dropout
    
    #try to use inbuilt test and predict functions
    #Getting results on Fact Retriever Module:
    for step, batch in enumerate(tqdm(val_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        dic = retriever.validation()
        sents = batch["examples"]
        #dic {"preds": preds, "labels": labels, "uids": uids}

    for step, batch in enumerate(tqdm(val_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
        batch_loss = questionClassificationModel.validation()
      #logging mechanism for loss
      #outputs a dictionary.. Need to deal with it
    f1 = f1_score(dic["labels"], dic["preds"], average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, dic["labels"], dic["labels"], sents

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

    retriever_config = yaml.load(open(constants.retriever_config))
    config = retriever_config["model"]["init_args"]
    retriever = RetrieverModel(config)
    retriever = retriever.to(device)

    #question classification config
    qc_config = yaml.load(open(constants.qc_config))
    config = qc_config["model"]["init_args"]
    questionClassificationModel = QuestionClassification(config)
    questionClassificationModel = questionClassificationModel.to(device)

    lr = args.lr
    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    #Note: make this parallel
    #Finetune the Question Classification Model
    for epoch in range(args.epochs):
        #this loop discard the .train() method
        train_loss = 0
        num_batches = 0
        optimizer = qc_config["model"]["optimizer"]["init_args"]
        optimizer.zero_grad()
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

    #Note: make this parallel
    #Finetune the Question Classification Model
    for epoch in range(args.epochs):
        #this loop discard the .train() method
        train_loss = 0
        num_batches = 0
        optimizer = retriever_config["model"]["optimizer"]["init_args"]
        optimizer.zero_grad()
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
    
    #Getting results on Fact Retriever Module(Retriever + Question Classification):
    #evaluate/predict using test set
    for step, batch in enumerate(tqdm(val_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
            batch_loss = retriever.validation()

    for step, batch in enumerate(tqdm(val_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE)):
            batch_loss = questionClassificationModel.validation()
            #logging mechanism for loss
            #outputs a dictionary.. Need to deal with it

    '''
    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            #Fact Retrieveing Module : Training stage.
            logits = retriever(batch) #forward or train?
            #get the loss and pass it to retirever to finetune it, as 
            b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        '''

def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


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
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
