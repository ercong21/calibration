import argparse
import random

import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from preprocessor import MLMPreprocessor
from processors.amazon import AmazonPolarityProcessor, AmazonStarProcessor, AmazonMultiProcessor
from processors.ag_news import AgNewsProcessor, AgNewsMultiProcessor
from processors.xnli import XnliProcessor, XnliMultiProcessor
from processors.pawsx import PawsxProcessor, PawsxMultiProcessor
from processors.yahoo import YahooProcessor
from processors.glue import ColaProcessor, Sst2Processor, MrpcProcessor, QnliProcessor, \
    QqpProcessor, RteProcessor, WnliProcessor
from processors.utils import InputExample
import log

logger = log.get_logger('root')

MULTI_TASKS = {
    "ag_news_multi", "amazon_reviews_multi", "xnli_multi", "pawsx_multi"
}

PROCESSORS = {
    'amazon_polarity': AmazonPolarityProcessor,
    'amazon_star': AmazonStarProcessor,
    'ag_news': AgNewsProcessor,
    'xnli': XnliProcessor,
    'pawsx': PawsxProcessor,
    'yahoo': YahooProcessor,
    'cola': ColaProcessor,
    'sst2': Sst2Processor,
    'mrpc': MrpcProcessor,
    'qnli': QnliProcessor,
    'qqp': QqpProcessor,
    'rte': RteProcessor,
    'wnli': WnliProcessor,
    'ag_news_multi': AgNewsMultiProcessor,
    'amazon_reviews_multi': AmazonMultiProcessor,
    'xnli_multi': XnliMultiProcessor,
    'pawsx_multi': PawsxMultiProcessor
}

def compute_metrics(preds, labels):
    return {
        "acc": (preds == labels).mean(),
        "num": len(preds),
        "correct": (preds == labels).sum(),
        "cm": confusion_matrix(labels, preds),
        'report': classification_report(labels, preds),
        'f1': f1_score(labels, preds, average='macro')
    }

def load_and_cache_dataset(args, preprocessor, processor=None, split=None, lang=None, examples=None):
    if processor:
        if args.multi_task:
            examples = processor.get_examples(args, split, lang)
        else:
            examples = processor.get_examples(args, split)

    features = []
    for example in tqdm(examples, desc='Creating input features from input examples'):
        input_features = preprocessor.get_input_features(example, labelled=True)
        features.append(input_features)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)
    all_idx = torch.tensor([int(f.idx.split('-')[-1]) for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_mlm_labels,
                                all_idx)
    return dataset

def evaluate(args, model, preprocessor, dataset, params=None):
    if args.save_logits:
        preds, out_label_ids = get_logits(args, model, preprocessor, dataset)
        with open(args.logits_save_path, 'wb') as f:
            pickle.dump((preds, out_label_ids), f)
    
    else:
        with open(args.logits_save_path, 'rb') as f:
            preds, out_label_ids = pickle.load(f)

    if params is not None:
        if args.calibration_strategy == 'penalty':
            assert len(params)==preds.shape[1], "size of params is wrong."
            for i in params.keys():
                preds[:, i] = preds[:, i] - params[i]

        elif args.calibration_strategy == 'transform':
            W, b = params
            calibrate_preds = None
            for logit, label in zip(preds, out_label_ids):
                calibrate_logit = np.matmul(W, np.expand_dims(logit, axis=-1)) + b
                if calibrate_preds is not None:
                    calibrate_preds = np.vstack((calibrate_preds, calibrate_logit.squeeze()))
                else:
                    calibrate_preds = calibrate_logit.squeeze()
            preds = calibrate_preds

    if args.calibration_strategy == 'cbm':
        # print(f'params: {preds.mean(axis=0)}')
        preds = preds / preds.mean(axis=0)

    predictions = np.argmax(preds, axis=1)
    return compute_metrics(predictions, out_label_ids)

def get_init_params(args, model, preprocessor):
    mask_ex_text = preprocessor.tokenizer.mask_token
    pseudo_ex = InputExample(guid='--3', text_a='', text_b='')
    prompt_ex_text = ''
    for part in preprocessor.pvp.get_parts(pseudo_ex):
        for token in part:
            if type(token)==tuple:
                prompt_ex_text += (token[0]+' ')
            else:
                prompt_ex_text += token
    examples = [InputExample(guid='--1', text_a=mask_ex_text), InputExample(guid='--2', text_a=prompt_ex_text)]
    dataset = load_and_cache_dataset(args, preprocessor, examples=examples)
    preds, out_label_ids = get_logits(args, model, preprocessor, dataset)

    if args.calibration_strategy == 'penalty':
        return {i: preds[0][i] for i in range(preds.shape[1])}
    
    if args.transform_context == 'mask':
        p_y = preds[0]
    elif args.transform_context == 'prompt':
        p_y = preds[1]
    elif args.transform_context == 'avg':
        p_y = np.mean(preds, axis=0)
    W = np.linalg.inv(np.identity(p_y.shape[0]) * p_y)
    b = np.zeros([p_y.shape[0], 1])
    return W, b

def train_params(args, model, preprocessor, dataset, initial_params=None):
    if args.calibration_strategy == 'penalty':
        return train_penalty(args, model, preprocessor, dataset, initial_params)
    elif args.calibration_strategy == 'transform':
        return train_transform(args, model, preprocessor, dataset, initial_params)

def train_transform(args, model, preprocessor, dataset, initial_params=None):
    preds, out_label_ids = get_logits(args, model, preprocessor, dataset)

    # path='logits/agnews_test.pk'
    # with open(path, 'rb') as f:
    #     preds, out_label_ids = pickle.load(f)

    if initial_params:
        W, b = initial_params
    else:
        W, b = get_init_params(args, model, preprocessor)
    
    for i in tqdm(range(args.penalty_train_epoch)):
        for logit, label in zip(preds, out_label_ids):
            transformed_logit = logit.copy()
            transformed_logit = np.matmul(W, np.expand_dims(logit, axis=-1)) + b
            transformed_logit = np.exp(transformed_logit) / np.sum(np.exp(transformed_logit), axis=0, keepdims=True)
            loss = -np.log(transformed_logit[label, 0])
            transformed_logit[label, 0] -= 1
            dW = np.dot(transformed_logit, np.expand_dims(logit, axis=0))
            W -= args.transform_train_lr * dW
            b -= args.transform_train_lr * transformed_logit
    
    return W, b

def train_penalty(args, model, preprocessor, dataset, initial_params=None):
    # if args.save_train_logits:
    #     preds, out_label_ids = get_logits(args, model, preprocessor, dataset)
    #     with open(args.logits_train_save_path, 'wb') as f:
    #         pickle.dump((preds, out_label_ids), f)
    # else:
    #     with open(args.logits_train_save_path, 'rb') as f:
    #         preds, out_label_ids = pickle.load(f)
    preds, out_label_ids = get_logits(args, model, preprocessor, dataset)

    # path='logits/agnews_test.pk'
    # with open(path, 'rb') as f:
    #     preds, out_label_ids = pickle.load(f)

    if initial_params:
        params = initial_params
    else:
        num_labels = preds.shape[1]
        initial_param = 1 / num_labels
        params = {i:initial_param for i in range(num_labels)}
    
    for i in tqdm(range(args.penalty_train_epoch)):
        for logit, label in zip(preds, out_label_ids):
            penalized_logit = logit.copy()
            for i in params.keys():
                penalized_logit[i] = logit[i] - params[i]
                pred = np.argmax(penalized_logit)
            if pred == label:
                continue
            else:
                # params[pred] += (logit[pred]-params[pred]) * args.penalty_train_lr
                # params[label] += (logit[label]-params[label]) * args.penalty_train_lr
                params[pred] +=  args.penalty_train_lr
                params[label] -=  args.penalty_train_lr
    
    return params
        

def get_logits(args, model, preprocessor, dataset):
    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    preds = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3],
                          "mlm_labels": batch[4], 'idx': batch[5]}
        labels = inputs['labels']
        indices = inputs['idx']

        with torch.no_grad():
            logits = mlm_eval_step(inputs, preprocessor, model)
            logits = F.softmax(logits, dim=1)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
    
    return preds, out_label_ids


def mlm_eval_step(batch, preprocessor, model):
    inputs = generate_default_inputs(batch)
    outputs = model(**inputs)

    return preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

def generate_default_inputs(batch):
    inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 
              'token_type_ids': batch['token_type_ids']}
    return inputs


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="bert-base-cased", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--task_name", default="amazon_polarity", type=str)
    parser.add_argument("--dataset_name", default="amazon_polarity", type=str)
    parser.add_argument("--pattern_id", default=1, type=int)
    parser.add_argument("--per_gpu_batch_size", default=8, type=int)
    parser.add_argument("--penalize", action="store_true")
    # parser.add_argument("--good_verbs", type=str, nargs='+', default=['good', 'perfect', 'fantastic', 'great', 'positive'])
    # parser.add_argument("--bad_verbs", type=str, nargs='+', default=['bad', 'awful', 'negative', 'terrible'])
    parser.add_argument("--num_train_sample", type=int, default=-1)
    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--train_split", type=str, default='train')
    parser.add_argument("--test_split", type=str, default='test')
    parser.add_argument("--penalty_train_epoch", type=int, default=0)
    parser.add_argument("--transform_train_epoch", type=int, default=0)
    parser.add_argument("--penalty_train_lr", type=float, default=0.0001)
    parser.add_argument("--transform_train_lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_logits", action='store_true', help="only save for the first time")
    parser.add_argument("--save_train_logits", action='store_true', help="only save for the first time")
    parser.add_argument("--result_dir", default='results/', type=str)
    parser.add_argument("--train_lang", default='en', type=str)
    parser.add_argument("--langs", default='af,co,en,eo,haw,hmn,ht,ig,jw,km,mi,mn,mt,my,ny,or,sm,sn,st,sw,ta,te,tl,ug,ur,uz,zu', type=str)
    parser.add_argument("--transform_context", default='mask', type=str, choices=['mask', 'prompt', 'avg'])
    parser.add_argument("--calibration_strategy", default='transform', type=str, choices=['penalty', 'transform', 'cbm'])


    args = parser.parse_args()

    args.multi_task = True if "multi" in args.task_name else False
    args.langs = args.langs.split(',')

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.logits_train_save_path = f"logits/train_{args.task_name}_p{args.pattern_id}.pk"
    args.result_file = f"{args.result_dir}{args.task_name}.csv"
    args.result_f1_file = f"{args.result_dir}{args.task_name}_f1.csv"

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    processor = PROCESSORS[args.task_name]()
    label_list = processor.get_labels()
    if args.penalty_train_epoch == 0 and args.num_train_sample != 0:
        args.penalty_train_epoch = max(int(5 // (args.penalty_train_lr * args.num_train_sample * len(label_list))), 1)
    if args.transform_train_epoch == 0 and args.num_train_sample != 0:
        args.transform_train_epoch = max(int(5 // (args.transform_train_lr * args.num_train_sample * len(label_list))), 1)

    logger.info('Parameters: {}'.format(args))
    
    # for good_verb in args.good_verbs:
    #     for bad_verb in args.bad_verbs:
    #         args.good_verb = good_verb
    #         args.bad_verb = bad_verb
    #         verbalizer_file = {0: [bad_verb], 1: [good_verb]}
    #         preprocessor = MLMPreprocessor(tokenizer, label_list, args.max_seq_length, args.task_name, args.pattern_id,
    #         verbalizer_file=verbalizer_file)


    #         loaded_dataset = load_and_cache_dataset(dataset, preprocessor, processor)

    #         results = evaluate(args, model, preprocessor, loaded_dataset)
    #         print(f"{args.good_verb}-{args.bad_verb}: {results['acc']}")
    preprocessor = MLMPreprocessor(tokenizer, label_list, args.max_seq_length, args.task_name, args.pattern_id, 
                                   model_name=args.model_name)
   
        
    params = None
    if args.penalize:
        # for cmb calibration, penalize doesn't need to be set 
        if args.num_train_sample == 0:
            params = get_init_params(args, model, preprocessor)
        else:
            train_dataset = load_and_cache_dataset(args, preprocessor, processor, args.train_split, lang=args.train_lang)
            # train_dataset = load_and_cache_dataset(args, preprocessor, processor, args.train_split) if args.save_train_logits else None

            params = train_params(args, model, preprocessor, train_dataset, params)
        logger.info(f"penalty params: {params}")

    # multilingual task
    if args.multi_task:
        accs = [args.num_train_sample]
        f1_scores = [args.num_train_sample]

        for lang in args.langs:
            args.logits_save_path = f"logits/logits_{args.model_name.split('-')[0]}_{args.calibration_strategy}/{args.task_name}_p{args.pattern_id}_{lang}.pk"
            test_dataset = load_and_cache_dataset(args, preprocessor, processor, args.test_split, lang=lang) \
            if args.save_logits else None
            results = evaluate(args, model, preprocessor, test_dataset, params)

            logger.info(f"******lang: {lang}******")
            logger.info(f"acc: {results['acc']}")
            logger.info(f"confusion matrix:\n{results['cm']}")
            logger.info(f"report:\n{results['report']}")
            logger.info(f"macro f1 score:\n{results['f1']}")

            accs.append(results['acc'])
            f1_scores.append(results['f1'])

        with open(args.result_file, 'a', newline='') as csvfile:
            with open(args.result_f1_file, 'a', newline='') as f1_file:
                writer = csv.writer(csvfile)
                writer_f1 = csv.writer(f1_file)
                writer.writerow(accs)
                writer_f1.writerow(f1_scores)


    # monolingual task
    else:     
        args.logits_save_path = f"logits/logits_{args.model_name.split('-')[0]}_{args.calibration_strategy}/{args.task_name}_p{args.pattern_id}.pk"
        # load test dataset
        test_dataset = load_and_cache_dataset(args, preprocessor, processor, args.test_split) if args.save_logits else None        

        results = evaluate(args, model, preprocessor, test_dataset, params)
        logger.info(f"acc: {results['acc']}")
        logger.info(f"confusion matrix:\n{results['cm']}")
        logger.info(f"report:\n{results['report']}")
        logger.info(f"macro f1 score:\n{results['f1']}")

        with open(args.result_file, 'a', newline='') as csvfile:
            with open(args.result_f1_file, 'a', newline='') as f1_file:
                writer = csv.writer(csvfile)
                writer_f1 = csv.writer(f1_file)
                writer.writerow([args.num_train_sample, results['acc']])
                writer_f1.writerow([args.num_train_sample, results['f1']])


if __name__ == "__main__":
    main()


    