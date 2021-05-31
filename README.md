# Improving Automated Evaluation of Open Domain Dialog via Diverse Reference Augmentation

Code and Data for our Findings of ACL 2021  paper titled 'Improving Automated Evaluation of Open Domain Dialog via Diverse Reference Augmentation. Varun Gangal \*, Harsh Jhamtani \*, Eduard Hovy, Taylor Berg-Kirkpatrick'


### Data
- Relevant original and augmented reference files in are present in 'ref_files/' in the required format
- Human ratings file: 'human_rating_correlation/mturk_rating_processed_output.csv'. Please consider citing [Gupta et al](https://github.com/prakharguptaz/multirefeval) if you use the human ratings file.



### Code
Code and script to compute metric correlations with human ratings can be found in 'human_rating_correlation/' directory


#### Requirements
- Python 3.7.5
- bert_score (0.3.7)
- [nlgeval](https://github.com/Maluuba/nlg-eval)(Accessed: December 2020)
- scipy 1.1.0




### Citation

```
@inproceedings{acl2021dialogeval, 
title={Improving Automated Evaluation of Open Domain Dialog via Diverse Reference Augmentation}, 
author={Gangal, Varun and Jhamtani, Harsh and Hovy, Ed and Berg-Kirkpatrick, Taylor}, 
booktitle={Findings of ACL}, 
year={2021} 
}
```
