#Evaluating Retriever's efficiency
import json
def retriever_eval(ori_file):
    with open(ori_file) as f:
        data_all = json.load(f)
    
    all_files_recall = 0.0
    
    #Go by each entry, and check if res_filename contains that id
    count_data = 0
    for data in data_all:
        true_sent_inds = data["qa"]["text_evidence"]
        true_table_inds = data["qa"]["table_evidence"]
        correct_text_ids = 0
        correct_table_ids = 0
        if "table_retrieved_topn" in data:
            pred_table_inds_topn = [i["ind"] for i in data["table_retrieved_topn"]]
            correct_table_ids = len(set(pred_table_inds_topn).intersection(set(true_table_inds))) 
        if "text_retrieved_topn" in data:
            pred_sent_inds_topn = [i["ind"] for i in data["text_retrieved_topn"]]
            correct_text_ids = len(set(pred_sent_inds_topn).intersection(set(true_sent_inds))) 
        all_files_recall += (correct_table_ids+correct_text_ids) / (len(true_table_inds) + len(true_sent_inds))
        print(correct_table_ids,correct_text_ids)
    resultant_recall = all_files_recall / len(data_all)
    return resultant_recall
