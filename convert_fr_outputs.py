import argparse
import collections
import json
import os
import sys
import random


### for single sent retrieved
# json_in is the output of the Fact Retriever module as JSON files
# json_out is the destination for top-n facts retrieved in plain text format (rather than logits)
# topn is the number of facts proposed by the retriever
# convert_train does return output; instead, it writes the "converted" output
#  to json_out. It converts Fact Retriever output into suitable input for the
#  reasoning model.
def convert_train(json_in, json_out, topn, max_len = 256):
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        try:
            gold_inds = []
            cur_len = 0
            # Gets all table/text evidence retrieved by the FR module
            #  table/text_retrieved_all are inserted into the json files
            #  by the fact retriever to denote the results the retriever
            #  got for that question/input pair
            table_retrieved = each_data["table_retrieved_all"]
            text_retrieved = each_data["text_retrieved_all"]
            all_retrieved = table_retrieved + text_retrieved
            
            # Selects the 'gold' supporting facts as annotated in the dataset
            #  that is, the facts that specifically pertain to the question
            gold_table_inds = each_data["qa"]["table_evidence"]
            gold_text_inds = each_data["qa"]["text_evidence"]
            
            # Collects plain text from original data that pertains to the
            #  retrieved facts
            for ind in gold_table_inds:
                gold_inds.append(ind)
                cur_len += len(each_data["table_description"][ind].split())

            for ind in gold_text_inds:
                gold_inds.append(ind)
                try:
                    cur_len += len(each_data["paragraphs"][ind].split())
                except:
                    continue

            # Counts all 'non-gold' facts from the dataset that were retrieved
            #  as negative samples (probably for training the retriever)
            false_retrieved = []
            for tmp in all_retrieved:
                if tmp["ind"] not in gold_inds:
                    false_retrieved.append(tmp)

            sorted_dict = sorted(false_retrieved, key=lambda kv: kv["score"], reverse=True)
            res_n = topn - len(gold_inds)
            
            other_cands = []
            # Adds 'negative' samples to the top-n facts fed to RoBERTa-large
            while res_n > 0 and cur_len < max_len:
                next_false_retrieved = sorted_dict.pop(0)
                # Disregards false retrieved under a certain score threshold
                #  prevents facts retrieved merely to get up to 'n' retrieved
                #  from biasing the output
                if next_false_retrieved["score"] < 0:
                    break

                if type(next_false_retrieved["ind"]) == int:
                    cur_len += len(each_data["paragraphs"][next_false_retrieved["ind"]].split())
                    other_cands.append(next_false_retrieved["ind"])
                    res_n -= 1
                else:
                    cur_len += len(each_data["table_description"][next_false_retrieved["ind"]].split())
                    other_cands.append(next_false_retrieved["ind"])
                    res_n -= 1
            
            # recover the original order in the document
            #  Sorts retrieved facts by order of appearance in the document
            input_inds = gold_inds + other_cands
            context = get_context(each_data, input_inds)
            each_data["model_input"] = context
            del each_data["table_retrieved_all"]
            del each_data["text_retrieved_all"]
        except:
            print(each_data["uid"])
    # json_out now holds the plain-text inputs to the reasoning module
    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

def convert_test(retriever_json_in, question_classification_json_in, json_out, topn, max_len = 256):
    with open(retriever_json_in) as f_in:
        data = json.load(f_in)
    
    with open(question_classification_json_in) as f_in:
        qc_data = json.load(f_in)

    qc_map = {}
    for example in qc_data:
        qc_map[example["uid"]] = example["pred"]

    for each_data in data:
        cur_len = 0
        # Gets all table/text evidence retrieved by the FR module
        #  table/text_retrieved_all are inserted into the json files
        #  by the fact retriever to denote the results the retriever
        #  got for that question/input pair
        table_retrieved = each_data["table_retrieved_all"]
        text_retrieved = each_data["text_retrieved_all"]
        all_retrieved = table_retrieved + text_retrieved

        cands_retrieved = []
        for tmp in all_retrieved:
            cands_retrieved.append(tmp)

        sorted_dict = sorted(cands_retrieved, key=lambda kv: kv["score"], reverse=True)
        res_n = topn
        
        other_cands = []

        # Similar to same section in convert_train above, however
        #  Simply collects the outputted facts from the retriever without
        #  distinguishing here between the gold facts and non-gold facts
        while res_n > 0 and cur_len < max_len:
            next_false_retrieved = sorted_dict.pop(0)
            if next_false_retrieved["score"] < 0:
                break

            if type(next_false_retrieved["ind"]) == int:
                cur_len += len(each_data["paragraphs"][next_false_retrieved["ind"]].split())
                other_cands.append(next_false_retrieved["ind"])
                res_n -= 1
            else:
                cur_len += len(each_data["table_description"][next_false_retrieved["ind"]].split())
                other_cands.append(next_false_retrieved["ind"])
                res_n -= 1
        
        # sorts facts by order of appearnace in the original document
        input_inds = other_cands
        context = get_context(each_data, input_inds)
        each_data["model_input"] = context

        each_data["qa"]["predicted_question_type"] = qc_map[each_data["uid"]]
        del each_data["table_retrieved_all"]
        del each_data["text_retrieved_all"]

    # json_out now holds the plain-text inputs to the reasoning module
    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

# This method is getting the individual sentences (or sentence
#  representations of table info) from a JSON entry and then outputting a list
#  of indices that can be used to index the sentences from the JSON entry
#  without having to search for them every time you want them. However, it does
#  not appear to be called as a helper in any other function in this file.
#  Might be called in another file this is imported to.
#  'each_data' is a JSON entry, 'input_inds' are the indices of the specific
#  sentences you want recovered from the entry for later use.
def get_context(each_data, input_inds):
    context = []
    table_sent_map = get_table_sent_map(each_data["paragraphs"])
    inds_map = {}
    for ind in input_inds:
        if type(ind) == str:
            # Gets actual input with index from table_sent_map
            table_ind = int(ind.split("-")[0])
            sent_ind = table_sent_map[table_ind]
            if sent_ind not in inds_map:
                inds_map[sent_ind] = [ind]
            else:
                if type(inds_map[sent_ind]) == int:
                    inds_map[sent_ind] = [ind]
                else:
                    inds_map[sent_ind].append(ind)
        else:
            if ind not in inds_map:
                inds_map[ind] = ind
    
    for sent_ind in sorted(inds_map.keys()):
        if type(inds_map[sent_ind]) != list:
            context.append(sent_ind)
        else:
            for table_ind in sorted(inds_map[sent_ind]):
                context.append(table_ind)
    
    return context

# Retrieves all the sentences of the input that are specifically reformatted
#  table information and outputs them as a mapping from sequential integers to
#  the index of the relevant sentence in the json entry-field 'paragraphs'
def get_table_sent_map(paragraphs):
    table_index = 0
    table_sent_map = {}
    for i, sent in enumerate(paragraphs):
        if sent.startswith("## Table "):
            table_sent_map[table_index] = i
            table_index += 1
    return table_sent_map


# Main uses the above helper functions to convert JSON files representing
#  retriever output into files that contain only the top-n facts for each data
#  entry, thereby enabling evaluation and finetuning.
if __name__ == '__main__':
    
    json_dir_in = "output/retriever_output"
    question_classification_json_dir_in = "output/question_classification_output"
    json_dir_out = "dataset/reasoning_module_input"
    os.makedirs(json_dir_out, exist_ok = True)
    
    topn, max_len = 10, 256
    
    mode_names = ["train", "test", "dev"]
    for mode in mode_names:
        json_in = os.path.join(json_dir_in, f"{mode}.json")
        question_classification_json_in = os.path.join(question_classification_json_dir_in, f"{mode}.json")
        json_out_train = os.path.join(json_dir_out, mode + "_training.json")
        json_out_inference = os.path.join(json_dir_out, mode + "_inference.json")

        if mode == "train":
            convert_train(json_in, json_out_train, topn, max_len)
        if mode == "dev":
            convert_train(json_in, json_out_train, topn, max_len)
            convert_test(json_in, question_classification_json_in, json_out_inference, topn, max_len)
        elif mode == "test":
            convert_test(json_in, question_classification_json_in, json_out_inference, topn, max_len)
        
        print(f"Convert {mode} set done")
        
