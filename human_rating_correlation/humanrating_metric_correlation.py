# coding: utf-8
# Simplified code to run on final reference files
import numpy as np
import json
import csv
import pickle
import argparse
from nlgeval import NLGEval
nlgeval = NLGEval(metrics_to_omit=['CIDEr']) 
import os.path
from os import path



parser = argparse.ArgumentParser()
parser.add_argument('--human_evals_data_file', default='mturk_rating_processed_output.csv')
parser.add_argument('--use_references_from', default=None)
parser.add_argument('--max_num_multi_response', default=-1, type=int)
parser.add_argument('--include_bert_score', default=False, action='store_true')
parser.add_argument('--dump_name', default=None, type=str) # dump_name='tmp/cometcontextv1.multi.pkl'
args = parser.parse_args()


use_references_from = args.use_references_from
# format of a needed json file
# {
#     "model": "hredf",
#     "context_id": "73_4",
#     "human_average_rating": "2.4",
#     "response": "well , i see .",
#     "prevgt": "then tell me something about your background .",
#     "all_references": [
#       "okay . what experience do you have ?",
#       "how many years of software engineering do you have ?",
#       "did you bring a resume ?",
#       "then tell me something about your background ."
#     ],
#     "context": "excuse me . i have an appointment with mr . li at nine . may i come in ?||||yes , come in please . i am mr . li . you must be my liu , right ?||||yes , i am my liu . thanks .||||i 'd like to start this interview with some questions . why do you think you are qualified for this position ?||||according to your advertisement , you want an experienced software engineer . i think my background meets the requirement of this position ."
#   },

    



data_metrics = json.load( open(use_references_from,'r') )
max_num_multi_response = args.max_num_multi_response    
if max_num_multi_response!=-1:
    for row in data:
        row['all_references'] = row['all_references'][:max_num_multi_response]

        
       
    
    
from bert_score import score
# compute nlgeval metrics    
def compute_metrics(data, include_bert_score=False):
    for j,row in enumerate(data):
        row['all_references'] = [v for v in row['all_references'] if len(v)>0]
        if len(row['all_references'])==0:
            row['all_references'] = ['default']
        row['metrics'] = nlgeval.compute_individual_metrics(ref=row['all_references'], hyp=row['response'])
        if include_bert_score:
            P_mul, R_mul, F_mul = score([row['response']], 
                                        [row['all_references']], 
                                        lang="en", 
                                        rescale_with_baseline=True)
            P_mul, R_mul, F_mul = P_mul.data.cpu().item(), R_mul.data.cpu().item(), F_mul.data.cpu().item()
            row['metrics']['bert_prec'] = P_mul
            row['metrics']['bert_rec'] = R_mul
            row['metrics']['bert_f1'] = F_mul

compute_metrics(data_metrics, include_bert_score=args.include_bert_score)




# compute correlation metrics    
import scipy
def compute_correlations(data, metric='Bleu_2', method_cnt_check=5, method=None):
    # todo: method_cnt_check
    vals_aut = []
    vals_hum = []
    for row in data:
        if method:
            if row['model'] != method:
                continue
        vals_aut.append(float(row['metrics'][metric]))
        vals_hum.append(float(row['human_average_rating']))
    pearsonr = scipy.stats.pearsonr(np.array(vals_aut),np.array(vals_hum))
    spearmanr = scipy.stats.spearmanr(np.array(vals_aut),np.array(vals_hum))
    kendall_tau = scipy.stats.kendalltau(np.array(vals_aut),np.array(vals_hum))
    #correlation, pvalue
    return { 'spearmanr':spearmanr, 'pearsonr':pearsonr, 'kendall_tau':kendall_tau }
    


metrics_of_interest = [
         'Bleu_1',
         'Bleu_2',
         'Bleu_3',
         'Bleu_4',
         'ROUGE_L',
         'METEOR',
         'SkipThoughtCS',
         'EmbeddingAverageCosineSimilairty'
]
if args.include_bert_score:
    metrics_of_interest.extend(['bert_prec','bert_rec'])
    
    
ret = {}
for k in metrics_of_interest:
    info = compute_correlations(data_metrics, metric=k, method_cnt_check=5)
    ret[k] = {
        'pearsonr':info['pearsonr'][0],
        'pearsonr_pvalue':info['pearsonr'][1],
        'spearmanr':info['spearmanr'].correlation,
        'spearmanr_pvalue':info['spearmanr'].pvalue,
        'kendall_tau':info['kendall_tau'].correlation,
        'kendall_tau_pvalue':info['kendall_tau'].pvalue
    }
    
if args.dump_name is not None:
    pickle.dump( data_metrics, open(args.dump_name,'wb'))
    

print(json.dumps(ret, indent=2))


    
