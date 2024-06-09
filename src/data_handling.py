'''
Script with functions for handling data of IC-XLT experiments

'''

import numpy as np
import string


def get_class_set(dataset):
    return sorted(list(set([i for x in dataset['label'] for i in x])))

def get_class_objects(dataset_train,dataset_test):
    '''    
    Input:
        - dataset_train: Dataset (train) object (transformers) 
        - dataset_test: Dataset (test) object (transformers) 

    Returns:
        - class_set: list of classes present in the dataset
        - lbl2id dictionary
        - id2lbl dictionary
    '''
    class_set_train = set(get_class_set(dataset_train))
    class_set_test = set(get_class_set(dataset_test))
    class_set = sorted(list(class_set_train.union(class_set_test)))
    lbl2id_class = {c:i for i,c in enumerate(class_set)}
    id2lbl_class = {v:k for k,v in lbl2id_class.items()}
    return class_set,lbl2id_class, id2lbl_class

def dataset_multilabel_extractor(
        label_list
):
    '''
    Returns multilabel entries joint with ',' (mainly used for ACD - no effect on MASSIVE)
    Input
        label_list : 2d (multilabel) label list => ['class_1', 'class_2']
        
    Return
        1D list with multilabel entries separated by commas => ['class_1, class_2']
    '''
    return [', '.join(y) for y in label_list]

def split_string(s, tokens_to_remove = ['</s>','<extra_id_0>','<extra_id_1>']):
    for tok_rm in tokens_to_remove:
        s = s.replace(tok_rm,'')
        
    delimiters = string.whitespace + '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' #same as string.punctuation but wihout "#" as it is part of ACD labels
    return ''.join(c if c not in delimiters else ' ' for c in s).split()


def extract_outputs_from_list(output_list,class_set):
    '''
    Extracts the labels from a list of outputs. Useful for multi-label predictions.
    '''
    output = []

    for out in output_list:
        tmp = []
        split_out = split_string(out)
        for substring in split_out:
        
            for c in class_set:
                if substring==c:
                    tmp.append(c)

        output.append(tmp)
    return output

def get_onehot_true_labels(
    class_list, #2d array of multi-label classes
    class_set,
    label2id_class,
    missing_labels = [], 
):

    output = np.zeros((len(class_list),len(class_set)))
    for i,lbls in enumerate(class_list):
        for lbl in lbls:
            if lbl not in missing_labels:
                output[i,label2id_class[lbl]] = 1
            else:
                pass #just ignore it
    return output


def get_kshot_dataset(
    dataset,
    k = 1,
    seed = 42):
    '''
    Function to reduce datasets to k samples per-label.
    For multi-label dataset it exactly selects k per label, avoiding repetitions.

    Input:
        dataset : Dataset object with 'text' and 'label' keys
        k = 1 : Number of shots per label to keep
        seed = 42 : seed for selecting data samples.
        
    Returns:
        kshot_dataset : reduced dataset

    '''
    
    
    if k==0: ## zero shot case
        
        kshot_dataset = dataset.select([])
        
        return kshot_dataset
    
    np.random.seed(seed) #fix seed to draw samples
    data_len = len(dataset['text'])
    idxs = np.random.permutation(data_len) 
    class_set = get_class_set(dataset)

    ##iterate over randomized index order and choose idxs until the k buckets are full
    
    valid_idxs = [] #these will be the ones left in the final reduced dataset
    lbl_bucket = {lbl: k for lbl in class_set}
        
    print(f'Selecting {k}-shots per label for train...')

    j = 0
    
    while j < len(idxs):
        
        i = idxs[j]
        skip = False

        if len(dataset['label'][i])==0: #skip empty labels
            
            skip = True
            
        else:
            
            for lbl in dataset['label'][i]:
                
                if lbl_bucket[lbl]<1:
                    skip = True
                    break
                    
        if not skip:
            
            valid_idxs.append(i)
            
            for lbl in [x for x in dataset['label'][i]]:
                
                lbl_bucket[lbl] -= 1

        if sum(lbl_bucket.values())<1:
            
            break
            
        j += 1
            
    kshot_dataset = dataset.select(valid_idxs)    
    dlen = len(kshot_dataset['text'])

    print(f'Length of reduced training dataset: {dlen}')
    
    return kshot_dataset