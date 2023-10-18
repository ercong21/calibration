# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" XNLI utils (dataset loading and evaluation) """

import logging
import os
import random

from transformers import DataProcessor
from .utils import InputExample
from datasets import load_dataset

logger = logging.getLogger(__name__)


class AmazonPolarityProcessor(DataProcessor):
    """Processor for the Amazon polarity dataset."""
    
    def __init__(self):
        self.dataset_name = 'amazon_polarity'
        self.dataset = load_dataset(self.dataset_name, split='test')
        self.dataset_size = len(self.dataset)

    def get_test_dataset(self, args):
        dataset = self.dataset.shuffle(args.seed).select(range(10000))
        test_set = dataset['content']
        labels = dataset['label']

        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        # size of test set is 10,000
        if split=='test':
            num_sample = args.num_test_sample
            dataset = self.dataset.shuffle(42).select(range(10000))
        else:
            num_sample = args.num_train_sample
            dataset = self.dataset.shuffle(args.seed).select(range(10000, self.dataset_size))

        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text = data_ex['content']
            label = data_ex['label']
            assert isinstance(text, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1 = [], []
            labels = list(set([e.label for e in examples]))
            for example in examples:
                if example.label==labels[0] and len(l0)<num_sample:
                    l0.append(example)
                elif example.label==labels[1] and len(l1)<num_sample:
                    l1.append(example)
                elif len(l0)==num_sample and len(l1)==num_sample:
                    break
            examples = l0+l1

        return examples

   
    def get_labels(self):
        """See base class."""
        return [0, 1]
    

class AmazonStarProcessor(DataProcessor):
    """Processor for the Amazon star dataset."""
    
    def __init__(self):
        self.dataset_name = 'amazon_reviews_multi'
    
    def get_test_dataset(self, args):
        dataset = load_dataset(self.dataset_name, 'en', split='test')
        test_set = dataset['review_body']
        labels = dataset['stars']
        labels = [star-1 for star in labels]
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split=='test':
            num_sample = args.num_test_sample
            dataset = load_dataset(self.dataset_name, 'en', split=split)
        else:
            num_sample = args.num_train_sample
            dataset = load_dataset(self.dataset_name, 'en', split='validation')

        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text = data_ex['review_body']
            label = data_ex['stars']-1
            assert isinstance(text, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2, l3, l4 = [], [], [], [], []
            labels = list(set([e.label for e in examples]))
            for example in examples:
                if example.label==labels[0] and len(l0)<num_sample:
                    l0.append(example)
                elif example.label==labels[1] and len(l1)<num_sample:
                    l1.append(example)
                elif example.label==labels[2] and len(l2)<num_sample:
                    l2.append(example)
                elif example.label==labels[3] and len(l3)<num_sample:
                    l3.append(example)
                elif example.label==labels[4] and len(l4)<num_sample:
                    l4.append(example)
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample and len(l3)==num_sample and len(l4)==num_sample:
                    break
            examples = l0+l1+l2+l3+l4

        return examples

   
    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]


class AmazonMultiProcessor(DataProcessor):
    """Processor for the Amazon reviews multilingual dataset."""
    
    def __init__(self):
        self.dataset_name = 'amazon_reviews_multi'

    def get_examples(self, args, split, lang):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split=='test':
            num_sample = args.num_test_sample
            dataset = load_dataset(self.dataset_name, lang, split=split)
        else:
            num_sample = args.num_train_sample
            dataset = load_dataset(self.dataset_name, 'en', split='validation')

        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text = data_ex['review_body']
            label = data_ex['stars']-1
            assert isinstance(text, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2, l3, l4 = [], [], [], [], []
            labels = list(set([e.label for e in examples]))
            for example in examples:
                if example.label==labels[0] and len(l0)<num_sample:
                    l0.append(example)
                elif example.label==labels[1] and len(l1)<num_sample:
                    l1.append(example)
                elif example.label==labels[2] and len(l2)<num_sample:
                    l2.append(example)
                elif example.label==labels[3] and len(l3)<num_sample:
                    l3.append(example)
                elif example.label==labels[4] and len(l4)<num_sample:
                    l4.append(example)
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample and len(l3)==num_sample and len(l4)==num_sample:
                    break
            examples = l0+l1+l2+l3+l4

        return examples

   
    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]


amazon_processors = {
    "amazon_polarity": AmazonPolarityProcessor,
    "amazon_star": AmazonStarProcessor
}

amazon_output_modes = {
    "amazon_polarity": "classification",
    "amazon_star": "classification",
}

amazon_tasks_num_labels = {
    "amazon_polarity": 2,
    "amazon_star": 5,
}