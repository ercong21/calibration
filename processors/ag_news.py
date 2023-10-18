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
""" ag news utils (dataset loading and evaluation) """

import logging
import os
import random

from transformers import DataProcessor
from .utils import InputExample
from datasets import load_dataset

logger = logging.getLogger(__name__)


class AgNewsProcessor(DataProcessor):
    """Processor for the Ag news dataset."""
    
    def __init__(self):
        self.dataset_name = 'ag_news'

    def get_test_dataset(self, args):
        dataset = load_dataset(self.dataset_name, split='test')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['text']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        dataset = load_dataset(self.dataset_name, split=split)
        num_sample = args.num_test_sample if split=='test' else args.num_train_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text = data_ex['text']
            label = data_ex['label']
            assert isinstance(text, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2, l3 = [], [], [], []
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
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample and len(l3)==num_sample:
                    break
            examples = l0+l1+l2+l3

        return examples

   
    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]
    

class AgNewsMultiProcessor(DataProcessor):
    """Processor for the multilingual Ag news dataset."""
    
    def __init__(self):
        self.dataset_name = 'ag_news_multi'

    def get_examples(self, args, split, lang):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        num_sample = args.num_test_sample if split=='test' else args.num_train_sample
        
        path = os.path.join('data/ag_news', lang+'.txt')
        examples = []
        idx = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                try:
                    text, label = line.split('\t')
                except:
                    print(idx, text)
                example = InputExample(guid=str(idx), text_a=text.strip(), label=int(label.strip()))
                idx += 1
                examples.append(example)

        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2, l3 = [], [], [], []
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
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample and len(l3)==num_sample:
                    break
            examples = l0+l1+l2+l3

        return examples

   
    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]


ag_news_processors = {
    "ag_news": AgNewsProcessor,
    "ag_news_multi": AgNewsMultiProcessor,
}

ag_news_output_modes = {
    "ag_news": "classification",
}

ag_news_num_labels = {
    "ag_news": 4,
}