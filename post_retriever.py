import json
#This file goes to post-training of retriever
'''
1. Evaluates the output of the model according to facts extracted
2. Adds the retrieved facts to the data files for PG Module to work on it.
'''
def combine_all_outputs(dicts):
    super_dict = {}
    for d in dicts:
        for k, v in d.items():  # d.items() in Python 3+
            super_dict.setdefault(k, []).append(v)
    return super_dict

def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

def retriever_evaluate(output_dict, ori_file, topn):
    '''
    save results to file. calculate recall
    batch_logits = logits for the given indices(each row=[_,score], columns=#text evidences)
    batch_filename_ids = file uids (uid used)
    batch_inds = evidence ids [indices]
    ori_file = original file
    topn = #number of top facts to be extracted
    '''
    output_dict = combine_all_outputs(output_dict)
    #Sample example of 1 pass of for loop [0.4,0.5,0.1] , uid=45, facts=[4,90,24]
    all_files_logits = output_dict["logits"]
    all_files_filename_ids = output_dict["filename_id"]
    all_files_evidence_inds = output_dict["ind"]
    
    res_filename = {}
    res_filename_inds = {}
    #Understand the format of logits!
    for this_logit, this_filename_id, this_ind in zip(all_files_logits, all_files_filename_ids, all_files_evidence_inds):
        
        #declaring lists for every uid
        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
            
        #assigning scores for every batch_index
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1].item(), #why 1 here?
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)

    #get all entries from train file 
    with open(ori_file) as f:
        data_all = json.load(f)
    
    #all_files_recall = 0.0
    
    #Go by each entry, and check if res_filename contains that id
    count_data = 0
    for data in data_all:
        this_filename_id = data["uid"]
        
        #Discarding files that are not in our results
        if this_filename_id not in res_filename:
            continue
        
        #since it is present in true fact, we add 1
        count_data += 1
        
        #get all the text evidence indices of this file
        this_res = res_filename[this_filename_id]
        
        #sorting them according to its score, with highest score first
        sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
        
        # sorted_dict = sorted_dict[:topn]
        
        #true evidence ids
        true_sent_inds = data["qa"]["text_evidence"]
        true_table_inds = data["qa"]["table_evidence"]

        # table rows
        pred_table_inds_topn = []
        pred_sent_inds_topn = []
        # all retrieved
        pred_table_inds = []
        pred_sent_inds = []
        
        for tmp in sorted_dict[:topn]:
            if type(tmp["ind"]) == str:
                pred_table_inds_topn.append(tmp)
            else:
                pred_sent_inds_topn.append(tmp)

        for tmp in sorted_dict:
            if type(tmp["ind"]) == str:
                pred_table_inds.append(tmp)
            else:
                pred_sent_inds.append(tmp)

        #correct_table_ids = len(set(pred_table_inds_topn).intersection(true_table_inds)) 
        #correct_text_ids = len(set(pred_sent_inds_topn).intersection(true_sent_inds)) 
        
        #all_files_recall += (correct_table_ids+correct_text_ids) / (len(true_table_inds) + len(true_sent_inds))

        #putting all the predictions in the file again,
        data["table_retrieved_all"] = pred_table_inds
        data["text_retrieved_all"] = pred_sent_inds
        data["table_retrieved_topn"] = pred_table_inds_topn
        data["text_retrieved_topn"] = pred_sent_inds_topn

    #res = all_files_recall / len(data_all)
    
    #res_message = f"Top {topn}: {res}\n"
    
    #return res, res_message
    write_predictions(data, ori_file[:-4]+"_post_eval.json")
    return None
