#This file goes to post-training of Program Generation
import sys,math
import collections
import numpy as np
import torch
from torch.nn import functional as F
from model.utils.utils import get_op_const_list


sys.path.insert(0, '../utils/')
max_program_length = 30

# provides probability of model outputted logits
def compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    #if no logits are present
    if scores == None:
        return []
    #Getting the max score of all logits
    
    #applying softmax
    probs = F.softmax(scores).tolist()
    '''
    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    '''
    return probs

#computes program from logits, returns program ids
def compute_prog_from_logits(logits, max_program_length):
    pred_prog_ids = []
    loss = 0
    for cur_step in range(max_program_length):
        cur_logits = logits[cur_step]
        cur_pred_softmax = compute_softmax(cur_logits)
        #getting the operation with maximum probability
        #why on logits?
        #cur_pred_token = np.argmax(cur_logits.cpu()) 
        #loss -= np.log(cur_pred_softmax[cur_pred_token])
        cur_pred_token = np.argmax(cur_pred_softmax.cpu()) #get to device 
        loss -=np.log(cur_pred_softmax[cur_pred_token])
        pred_prog_ids.append(cur_pred_token)
        if cur_pred_token == 0:
            break
    return pred_prog_ids, loss

#Makes a program from program ids and other stuff
def indices_to_prog(program_indices, numbers, number_indices,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    #iterating over program indices and putting values into a prog list
    for prog_id in program_indices:
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog

#saves the program to the json file for further usage
def compute_predictions(all_examples, all_features, all_results, n_best_size):
    
    """Computes final predictions based on logits."""

    op_list, const_list = get_op_const_list()
    const_list_size = len(const_list)
    op_list_size = len(op_list)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature["example_index"]].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "logits"
        ])

    all_predictions = collections.OrderedDict()
    all_predictions["pred_programs"] = collections.OrderedDict()
    all_predictions["ref_programs"] = collections.OrderedDict()
    all_nbest = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        if example_index not in example_index_to_features:
            continue
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            if feature["unique_id"] not in unique_id_to_result:
                continue
            result = unique_id_to_result[feature["unique_id"]]
            logits = result.logits
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    logits=logits))

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", "options answer program_ids program")

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            program = example.program
            pred_prog_ids, loss = compute_prog_from_logits(pred.logits,
                                                           max_program_length)
            #logger for loss here --
            pred_prog = indices_to_prog(pred_prog_ids,
                                        example.numbers,
                                        example.number_indices,
                                        op_list, op_list_size,
                                        const_list, const_list_size
                                        )
            nbest.append(
                _NbestPrediction(
                    options=example.options,
                    answer=example.answer,
                    program_ids=pred_prog_ids,
                    program=pred_prog))

        # assert len(nbest) >= 1
        if len(nbest) == 0:
            continue
        
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["id"] = example.id
            output["options"] = entry.options
            output["ref_answer"] = entry.answer
            output["pred_prog"] = [str(prog) for prog in entry.program]
            output["ref_prog"] = example.program
            output["question_tokens"] = example.question_tokens
            output["numbers"] = example.numbers
            output["number_indices"] = example.number_indices
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions["pred_programs"][example_index] = nbest_json[0]["pred_prog"]
        all_predictions["ref_programs"][example_index] = nbest_json[0]["ref_prog"]
        all_nbest[example_index] = nbest_json

    return all_predictions, all_nbest
