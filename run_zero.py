import random

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
import time
import argparse
from processors.amazon import AmazonPolarityProcessor, AmazonStarProcessor
from processors.ag_news import AgNewsProcessor
from processors.xnli import XnliProcessor
from processors.pawsx import PawsxProcessor
from processors.yahoo import YahooProcessor
from prompt_labels import ID2LABELS, PATTERNS

PROCESSORS = {
    'amazon_polarity': AmazonPolarityProcessor,
    'amazon_star': AmazonStarProcessor,
    'ag_news': AgNewsProcessor,
    'xnli': XnliProcessor,
    'pawsx': PawsxProcessor,
    'yahoo': YahooProcessor
}


def evaluate(zero_ppl, test_set, labels, id2label):
    candidate_labels = list(id2label.values())
    label2id = {label: id for id, label in id2label.items()}
    results = zero_ppl(test_set, candidate_labels, multi_label=False, batch_size=128)
    predictions = np.array([label2id[result['labels'][0]] for result in results])
    labels = np.array(labels)
    return (predictions==labels).mean()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path_or_name", default="../zero-classify/models/bert-base-cased-nli", 
                        type=str, help="The pretrained model used for the zero-shot classification pipeline")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--task_name", default='ag_news', type=str)
    parser.add_argument("--num_sample", default=-1, type=int)
    parser.add_argument("--device",default=0, type=int)
    parser.add_argument("--pattern_id",default=0, type=int)

    args = parser.parse_args()

    model_path = args.model_path_or_name

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.model_max_length = args.max_seq_length
    model.config.label2id = {
        "contradiction": 2,
        "entailment": 0,
        "neutral": 1
    }

    random.seed(args.seed)

    # load test set
    processor = PROCESSORS[args.task_name]()
    test_set, labels = processor.get_test_dataset(args)

    print(args)

    zero_classify = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, device=args.device, batch_size=args.batch_size)

    # get id2label
    id2label = ID2LABELS[args.task_name]
    pattern = PATTERNS[args.task_name]
    id2prompt_label = {id: pattern(label, args.pattern_id) for id, label in id2label.items()}

    start = time.time()  
    acc = evaluate(zero_classify, test_set, labels, id2prompt_label)
    print('Acc: %.4f' % acc) 
    print('Processing time: %.4fs' % (time.time()-start))

if __name__ == "__main__":
    main()
