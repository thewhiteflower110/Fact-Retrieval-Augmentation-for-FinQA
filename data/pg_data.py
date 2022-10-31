# Span generation Dataset and dataloader!!
import json
from transformers import AutoTokenizer
import constants
import re
import torch
from typing import Dict, Iterable, List, Any, Optional, Union
from torch.utils.data import Dataset
from models.utlis.utils import get_op_const_list

class ProgramGenerationDataset(Dataset):
    def __init__(self,data_file,transformer_model_name,is_training,entity_name: str = "question_type"):
        self.data_file = data_file
        self.data_all = self.read()
        self.transformer_model_name = transformer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.max_seq_length = constants.max_seq_length #### add to constant as 30
        self.op_list = constants.op_list
        self.const_list = constants.const_list
        self.const_list_size = len(self.const_list)
        self.op_list_size = len(self.op_list)
        self.max_program_length = constants.max_program_length ### add to constants
        self.is_training=is_training
        self.entity_name = entity_name
        self.data_all = self.read()
        self.instances = self.get_features()

    def read(self):
        with open(self.data_file) as f:
            data_all = json.load(f)
        return data_all

    def get_features(self):    
        examples = []
        features_all=[]
        for example in self.data_all:
            features = self.convert_example_to_feature(example, self.tokenizer, self.entity_name)
            features_all.append(features)
            if example:
                examples.append(example)
        return features_all

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def str_to_num(self,text):
        text = text.replace("$","")
        text = text.replace(",", "")
        text = text.replace("-", "")
        text = text.replace("%", "")
        try:
            num = float(text)
        except ValueError:
            if "const_" in text:
                text = text.replace("const_", "")
                if text == "m1":
                    text = "-1"
                num = float(text)
            else:
                num = "n/a"
        return num

    #tokenizes the given text and takes care of special tokens
    def tokenize(self,tokenizer, text, apply_basic_tokenization=False):
        """Tokenizes text, optionally looking up special tokens separately.
        Args:
        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
        text: text to tokenize
        apply_basic_tokenization: If True, apply the basic tokenization. If False,
            apply the full tokenization (basic + wordpiece).
        Returns:
        tokenized text.
        A special token is any text with no spaces enclosed in square brackets with no
        space, so we separate those out and look them up in the dictionary before
        doing actual tokenization.
        """

        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

        #bert basic tokenizer 
        tokenize_fn = tokenizer.tokenize
        if apply_basic_tokenization:
            tokenize_fn = tokenizer.basic_tokenizer.tokenize

        #tokenize the usual text, and for special tokens, use them if present or use UNK
        tokens = []
        for token in text.split(" "):
            if _SPECIAL_TOKENS_RE.match(token):
                #put the special tokens as it
                if token in tokenizer.get_vocab():
                    tokens.append(token)
                # put the special tokens as UNKs
                else:
                    tokens.append(tokenizer.unk_token)
            else:
                tokens.extend(tokenize_fn(token))
        return tokens

    #tokenize the original program
    def program_tokenization(self,original_program):
        original_program = original_program.split(',')
        program = []
        for tok in original_program:
            tok = tok.strip()
            cur_tok = ''
            for c in tok:
                if c == ')':
                    if cur_tok != '':
                        program.append(cur_tok)
                        cur_tok = ''
                cur_tok += c
                if c in ['(', ')']:
                    program.append(cur_tok)
                    cur_tok = ''
            if cur_tok != '':
                program.append(cur_tok)
        program.append('EOF')
        return program

    #converts program tokens to indices using op_list, const_list
    #and the numbers from the existing facts of that exaample
    def prog_token_to_indices(self,prog, numbers, number_indices):
        prog_indices = []
        for i, token in enumerate(prog):
            if token in self.op_list:
                prog_indices.append(self.op_list.index(token))
            elif token in self.const_list:
                prog_indices.append(self.op_list_size + self.const_list.index(token))
            else:
                if token in numbers:
                    cur_num_idx = numbers.index(token)
                else:
                    cur_num_idx = -1
                    for num_idx, num in enumerate(numbers):
                        if self.str_to_num(num) == self.str_to_num(token) or (self.str_to_num(num) != "n/a" and self.str_to_num(num) / 100 == self.tr_to_num(token)):
                            cur_num_idx = num_idx
                            break
                        
                if cur_num_idx == -1:
                    return None
                prog_indices.append(self.op_list_size + self.const_list_size +
                                    number_indices[cur_num_idx])
        return prog_indices

    #concatenating each sentence with its question and converting to model facourable inputs
    def concatenating(self,tokenizer,example,cls_token, sep_token):
        '''
        single pair of question, context, label feature
        ##  DONT KNOW WHAT IS cls token and sep token!!
        '''
        features=[]
        question_tokens = example["question_tokens"]
        if len(question_tokens) >  self.max_seq_length - 2:
            print("too long")
        question_tokens = question_tokens[:self.max_seq_length - 2]

        tokens = [cls_token] + question_tokens + [sep_token]
        #initialized to 0
        segment_ids = [0] * len(tokens)
        
        #getting ids of token and initializing the mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        #find what is number_indices?
        for ind, offset in enumerate(example["number_indices"]):
            if offset < len(input_mask):
                input_mask[offset] = 2
            else:
                if self.is_training == True:
                    return features

        #converting to max_seq_lengtj
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)
            
        number_mask = [tmp - 1 for tmp in input_mask]
        for ind in range(len(number_mask)):
            if number_mask[ind] < 0:
                number_mask[ind] = 0
        option_mask = [1, 0, 0, 1] + [1] * (len(self.op_list) + len(self.const_list) - 4)
        option_mask = option_mask + number_mask
        option_mask = [float(tmp) for tmp in option_mask]

        for ind in range(len(input_mask)):
            if input_mask[ind] > 1:
                input_mask[ind] = 1

        numbers = example["numbers"]
        number_indices = example["number_indices"]
        program = example["program"]

        
        if program is not None and self.is_training:
            program_ids = self.prog_token_to_indices(program, numbers, number_indices)
            if not program_ids:
                return None
            
            program_mask = [1] * len(program_ids)
            program_ids = program_ids[:self.max_program_length]
            program_mask = program_mask[:self.max_program_length]
            if len(program_ids) < self.max_program_length:
                padding = [0] * (self.max_program_length - len(program_ids))
                program_ids.extend(padding)
                program_mask.extend(padding)
        else:
            program = ""
            program_ids = [0] * self.max_program_length
            program_mask = [0] * self.max_program_length
        
        this_input_features = {
            "id": example.id,
            "unique_id": -1,
            "example_index": -1,
            "tokens": tokens,
            "question": example.original_question,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "option_mask": option_mask,
            "segment_ids": segment_ids,
            "options": example.options,
            "answer": example.answer,
            "program": program,
            "program_ids": program_ids,
            "program_weight": 1.0,
            "program_mask": program_mask
        }
        return this_input_features

    def convert_example_to_feature(self,example, tokenizer, entity_name):
        if example["qa"][entity_name] != "arithmetic":
            return None
        #getting the extracted facts by retriever model
        context = ""
        for idx in example["model_input"]:
            if type(idx) == int:
                context += example["paragraphs"][idx][:-1]
                context += " "

            else:
                context += example["table_description"][idx][:-1]
                context += " "
        
        question = example["qa"]["question"]
        this_id = example["uid"]

        original_question = question + " " + tokenizer.sep_token + " " + context.strip()

        options = example["qa"]["answer"] if "answer" in example["qa"] else None
        answer = example["qa"]["answer"] if "answer" in example["qa"] else None

        original_question_tokens = original_question.split(' ')
        numbers = []
        number_indices = []
        question_tokens = []

        #extracting all the numbers from question and putting it to numbers and their indices in number_indices
        for i, tok in enumerate(original_question_tokens):
            num = self.str_to_num(tok)
            if num is not None:
                if num != "n/a":
                    numbers.append(str(num))
                else:
                    numbers.append(tok)
                number_indices.append(len(question_tokens))
                if tok and tok[0] == '.':
                    numbers.append(str(self.str_to_num(tok[1:])))
                    number_indices.append(len(question_tokens) + 1)
            tok_proc = self.tokenize(tokenizer, tok)
            question_tokens.extend(tok_proc)


        #converting the original program to tokens
        original_program = example["qa"]['program'] if "program" in example["qa"] else None
        if original_program:
            program = self.program_tokenization(original_program)
        else:
            program = None

        example={"id":this_id,
            "original_question":original_question,
            "question_tokens":question_tokens,
            "options":options,
            "answer":answer,
            "numbers":numbers,
            "number_indices":number_indices,
            "original_program":original_program,
            "program":program}

        #Dont know where these come from cls_token, sep_token
        return self.concatenating(tokenizer,example, cls_token, sep_token)


