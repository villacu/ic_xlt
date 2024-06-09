'''

Script with functions for IC-XLT and experiments

'''

from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
from datasets import Dataset
import numpy as np
from peft import PeftModel, LoraConfig,TaskType, get_peft_model
from transformers import Trainer, GenerationConfig
from sklearn.metrics import f1_score

from src.data_handling import dataset_multilabel_extractor, extract_outputs_from_list, get_onehot_true_labels


## context-related functions

def create_context(
    text_list,
    label_list,
    idxs_in_order,
    prepend_text,
    prepend_labels,
    intercontext_char,
    lower_labels = False,):
    '''
    Auxiliar function for generating context
    Input:
        text_list : list of texts in context
        label_list : list of labels in context
        idxs_in_order : index ordering for context
        prepend_text : String to prepend to texts in context
        prepend_labels : String to prepend to labels in context
        intercontext_char : character between each of the context samples
    '''
    
    if type(label_list[0])==list: #2d list
        #print('2d list')
        labels = label_list

    else: #1d - output if unsqueeze = True
        #print('1d list')
        labels = [[y] for y in label_list] ##also converts to 2d to use same return

    if lower_labels:
        labels = [[x.lower() for x in y] for y in labels]
    
    return intercontext_char.join([prepend_text+text_list[k]+prepend_labels+', '.join(labels[k]) for k in idxs_in_order]) + intercontext_char

def create_ict_context(
    text_list,
    label_list,
    n_context = 10,
    seed = 42,
    prepend_text = 'Text: ',
    prepend_labels = ' Labels: ',
    intercontext_char = '\n',
    append_labels_in_main =  True,
    lower_labels = False
    ):
    '''
    Creates samples with random in-context examples to be used in in-context tuning (ICT)
    Input:
        text_list : list of text samples
        label_list : lists of label samples, should be paired with text samples (should be 2d in case of multilabel)
        n_context : number of in-context examples to add
        seed : seed for randomness in in context examples
        prepend_text : String to prepend to texts in context
        prepend_labels : String to prepend to labels in context
        append_labels_in_main : Whether to prepend the 'Labels:' string to the main text
        intercontext_char : character between each of the context samples
    Returns:
        ict_text_list
    '''

    data_length = len(text_list)

    assert data_length==len(label_list)

    ## fix seed
    np.random.seed(seed)

    ## generate ICT text list
    ict_text_list = []
    for i in range(data_length):

        ## select indices for -random- context
        idxs_in_order = [i]
        while i in idxs_in_order:
            idxs_in_order = np.random.choice(range(data_length),n_context,replace=False)
        
        ## create context
        sample = create_context(
            text_list = text_list,
            label_list = label_list,
            idxs_in_order = idxs_in_order,
            prepend_text = prepend_text,
            prepend_labels = prepend_labels,
            intercontext_char = intercontext_char,
            lower_labels = lower_labels
        )

        ## add sample with in context to final list

        sample += prepend_text+text_list[i]

        if append_labels_in_main:
            sample += prepend_labels
        
        ict_text_list.append(sample)
    
    return ict_text_list

def create_icl_dataset(
    dataset_test,
    dataset_train,
    prepend_text = 'Text: ',
    prepend_labels = ' Labels: ',
    intercontext_char = '\n',
    seed_ordering = 42,
    lower_labels = False,
    ):
    
    '''
    Creates dataset with with in-context examples to be used in in-context learning (ICL)
    
    Input:
        dataset_test : dataset with test samples, the resulting icl dataset will have the same size
        dataset_train : dataset with (few-shot) training samples to be prepended to each test entry. All dataset_train samples 
                        used so it should be a few-shot dataset (<48 samples if possible)
        
        prepend_text : String to prepend to texts in context
        prepend_labels : String to prepend to labels in context
        intercontext_char : character between each of the context samples
        seed_ordering : seed for randomness in context ordering
        
    Returns:

        dataset_icl : dataset_icl['text'] includes prepended in-context examples

    '''

    test_length = len(dataset_test['text'])
    train_length = len(dataset_train['text'])

    ### set random seed
    np.random.seed(seed_ordering)
    

        
    index_ordering = np.random.permutation(train_length)

    context = create_context(
            text_list = dataset_train['text'],
            label_list = dataset_train['label'],
            idxs_in_order = index_ordering,
            prepend_text = prepend_text,
            prepend_labels = prepend_labels,
            intercontext_char = intercontext_char,
            lower_labels = lower_labels
            )
    
    
    ## prepend context demonstrations to all samples in the test set
    
    text_with_context = [context + prepend_text + dataset_test['text'][i] + prepend_labels 
                         for i in range(test_length)]
    
        
    return Dataset.from_dict({'text':text_with_context,'label':dataset_test['label']})

## preprocessing / tokenization

def preprocess_function(
        sample, 
        tokenizer, 
        ict_n = None,
        ict_seed = 42,
    ):
    '''
    This function preprocess and tokenizes the input data for the mT5 model.
    The preprocessing includes generating the context examples and prepending them to the context

    Inputs:
        - sample : dataset
        - tokenizer : tokenizer 
        - ict_n : number of examples prepended to context (M variable in the paper),
        - ict_seed : seed for the context examples (since they are randomly drawn from the training data),
    
    Returns
        - model_inputs : preprocessed and tokenized data
    
    '''

    if ict_n is None or ict_n==0:

        max_length = 128

        text_list = sample["text"]  

    else:
        
        max_length = 512 #longer max_length for ICT

        text_list = create_ict_context(
                        sample["text"],
                        sample["label"],
                        n_context = ict_n,
                        seed = ict_seed,
                        )
    
    # tokenize inputs
    model_inputs = tokenizer(
        text_list, 
        max_length = max_length, 
        padding = "max_length", 
        truncation = True)

    # tokenize targets 
    sample_labels = dataset_multilabel_extractor(
                        sample["label"],
                        # ACD_EA_index = None
                        )
    
    # print(f'sample labels: {sample_labels[0]}')

    labels = tokenizer(
        text_target = sample_labels, 
        max_length = 128, 
        padding = "max_length", 
        truncation = True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

## functions for evaluation

def compute_metrics(
    predicted_labels,
    truth_labels,
    class_set,
    lbl2id_class
):
    '''
    Input:
        predicted_labels : predicted classes per sample (separated by commas if multilabel)
        truth_labels : correct classes per sample (separated by commas if multilabel)
        class_set : class_set
        lbl2id_class : lbl2id_class
    Returns:
        metrics : dictionary with F1-micro metric and F1 metric per labels
    
    '''
    
    # retrieve labels from the model's text output
    
#     preds_clean = extract_outputs_from_list(predicted_labels, class_set)
#     labels_clean = extract_outputs_from_list(truth_labels, class_set)

    preds_clean = predicted_labels#extract_outputs_from_list(, class_set)
    labels_clean = truth_labels #extract_outputs_from_list(truth_labels, class_set)
    
    # convert to onehot to compute metrics
    y_true = get_onehot_true_labels(
            labels_clean, #2d array of multi-label classes
            class_set,
            lbl2id_class)
        
    y_pred = get_onehot_true_labels(
        preds_clean, #2d array of multi-label classes
        class_set,
        lbl2id_class)

    metrics = {
        'f1_score_micro':f1_score(y_true,y_pred,average='micro'),
        'f1_score_macro':f1_score(y_true,y_pred,average='macro'),
        }


    for lbl_idx in range(y_true.shape[1]):
        metric_tag = 'f1_' + class_set[lbl_idx]
        metrics[metric_tag] = f1_score(y_true[:,lbl_idx],y_pred[:,lbl_idx])

    return metrics

## functions for training

def train_lora(
    base_model,
    peft_training_args,
    dataset_train,
    
    lora_checkpoint = None,
    lora_config = None,
    ):
    
    '''
    Trains the LoRA adapter on top of the base model
    
    Input:
        base_model : base seq2seq model to load LoRA on
        peft_training_args : TrainingArguments object
        dataset_train : tokenized dataset for training

        lora_checkpoint = None : checkpoint with previously trained LoRA (to continue fine-tuning)
        lora_config = None : LoraConfig object with LoRA adapter hyperparamters
    Returns:
        model : model with training LoRA
    '''

    try:
        ## load existing adapter
        
        model = PeftModel.from_pretrained(
            base_model,
            lora_checkpoint, # load checkpoint if provided 
            is_trainable=True) 
        
        print('\n > Existing LoRA loaded\n\n')
    
    except:
        ## load new adapter
        
        if lora_config is None:
            
            lora_config = LoraConfig(
                r = 16, 
                lora_alpha = 32, 
                target_modules = ['q', 'v'],
                lora_dropout = 0.1, 
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM, 
                )
        
        model = get_peft_model(base_model, lora_config)
        
        print('\n > New LoRA loaded\n\n')

    trainer = Trainer(
        model = model,
        args = peft_training_args,
        train_dataset = dataset_train
    )
    
    ## train
    trainer.train()
    
    return model
    
## functions for generation/evaluation

def generate_predictions_t5(
    model,
    tokenizer,
    text_list,
    batch_size = 8,
    max_new_tokens = 64,
):
    '''
    Function to generate output labels from T5 model.
    
    This kind of simulates the trainer.predict() option (since traditional Trainer has no option of 
    predict_with_generate and Seq2SeqTrainer does not work with PEFT at the moment of this code.)
    
    Input:
        model : model with trained LoRA
        tokenizer : model tokenizer
        text_list : text to predict (not tokenized)
        batch_size = 8: batch size for generating predictions
        max_new_tokens = 8: max number of new tokens for generated labels
    Returns:
        sequence_outputs : Tokenized generated sequence to be fed to compute metrics function
    
    '''
    
#     print(f'Batch size in generation set to {batch_size}')
    
    sequence_outputs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for i in tqdm(range(0,len(text_list),batch_size)):
        
        prompt_subset = text_list[i:i+batch_size]
        
        ## tokenize inputs and send to device
        inputs = tokenizer(
            prompt_subset, 
            return_tensors="pt",
            pad_to_max_length = True,
        ).input_ids.to(model.device)
        
        ## generate predictions
        outputs = model.generate(
            input_ids = inputs, 
            generation_config = GenerationConfig(
                max_new_tokens = max_new_tokens, 
                num_beams = 1))
        
        ## pad if is less than 
        if outputs.shape[-1] < (max_new_tokens + 1):
            diff = (max_new_tokens + 1) - outputs.shape[-1]
            ## pad to the left with tokenizer
            outputs = F.pad(outputs, (0, diff), "constant", tokenizer.pad_token_id)
        
        sequence_outputs += outputs
    
    ## stack the predicted samples
    sequence_outputs = torch.stack(sequence_outputs) 
    return sequence_outputs

def run_inference(
    model,
    tokenizer,
    test_texts,
    class_set,
    batch_size = 8,
    max_new_tokens = 64
):
    '''
    Predicts a set of input texts and return the predicted labels
    
    Input:
        model : base model with loaded LoRA
        tokenizer : base model tokenizer
        config : config file (can be the same used for training)
        test_texts : (already processed) list of test prompts (with or withour context)
        lbl2id_class : for extracting the ids from the predictions, if is set to None, predictions are 
                        returned as strings

    Returns:
        predicted_labels : lists with the predicted labels
    '''

    ## get predictions from input text
    output_preds = generate_predictions_t5(
        model = model,
        tokenizer = tokenizer,
        text_list = test_texts,
        batch_size = batch_size, #config_xeval['batch_size_generation'],
        max_new_tokens = max_new_tokens,
    )

    ## decode predictions and extract predictions
    decoded_predictions = tokenizer.batch_decode(output_preds, skip_special_tokens=True)
    clean_predictions = extract_outputs_from_list(decoded_predictions, class_set)
    
    return clean_predictions

