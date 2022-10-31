#Evaluating Retriever's efficiency
import json
def retriever_eval(ori_file):
    with open(ori_file) as f:
        data_all = json.load(f)
    
    all_files_recall = 0.0
    
    #Go by each entry, and check if res_filename contains that id
    count_data = 0
    for data in data_all:
        if "table_retrieved_topn" in data:
            pred_table_inds_topn = data["table_retrieved_topn"]
            pred_sent_inds_topn = data["text_retrieved_topn"] 
            true_sent_inds = data["qa"]["text_evidence"]
            true_table_inds = data["qa"]["table_evidence"]
            correct_table_ids = len(set(pred_table_inds_topn).intersection(true_table_inds)) 
            correct_text_ids = len(set(pred_sent_inds_topn).intersection(true_sent_inds)) 
            all_files_recall += (correct_table_ids+correct_text_ids) / (len(true_table_inds) + len(true_sent_inds))
    
    resultant_recall = all_files_recall / len(data_all)
    return resultant_recall
