
mkdir tmp
mkdir results


############################################################ SINGLE and MULTI

# multi reference
CUDA_VISIBLE_DEVICES=1 python humanrating_metric_correlation.py --include_bert_score --use_references_from ../ref_files/multiref.json | tee results/multiref.log
# single reference
CUDA_VISIBLE_DEVICES=1 python humanrating_metric_correlation.py --include_bert_score --use_references_from ../ref_files/singleref.json | tee results/singleref.log

############################################################ SCARCE

# scarce-multi reference
CUDA_VISIBLE_DEVICES=1 python humanrating_metric_correlation.py --include_bert_score --use_references_from ../ref_files/scarce.multiref.json | tee results/scarce.multiref.log
# scarce-single reference
CUDA_VISIBLE_DEVICES=1 python humanrating_metric_correlation.py --include_bert_score --use_references_from ../ref_files/scarce.singleref.json | tee results/scarce.singleref.log




