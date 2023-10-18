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


class YahooProcessor(DataProcessor):
    """Processor for the Amazon polarity dataset."""
    
    def __init__(self):
        self.dataset_name = 'yahoo_answers_topics'
        self.dataset = load_dataset(self.dataset_name, split='test')
        self.dataset_size = len(self.dataset)
    
    def get_test_dataset(self, args):
        dataset = self.dataset.shuffle(args.seed).select(range(10000))
        test_set = [text_a+'\ '+text_b for text_a, text_b in zip(dataset['question_title'], dataset['best_answer'])]
        labels = dataset['topic']
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        # size of test set is 10,000
        if split=='train':
            num_sample = args.num_train_sample
            dataset = self.dataset.shuffle(args.seed).select(range(10000))
        else:
            num_sample = args.num_test_sample
            dataset = self.dataset.shuffle(42).select(range(10000, self.dataset_size))

        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['question_title']
            text_b = data_ex['best_answer']
            label = data_ex['topic']
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9 = [], [], [], [], [], [], [], [], [], []
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
                elif example.label==labels[5] and len(l5)<num_sample:
                    l5.append(example)
                elif example.label==labels[6] and len(l6)<num_sample:
                    l6.append(example)     
                elif example.label==labels[7] and len(l7)<num_sample:
                    l7.append(example)
                elif example.label==labels[8] and len(l8)<num_sample:
                    l8.append(example)
                elif example.label==labels[9] and len(l9)<num_sample:
                    l9.append(example)                                                    
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample and len(l3)==num_sample and len(l4)==num_sample and len(l5)==num_sample and len(l6)==num_sample and len(l7)==num_sample and len(l8)==num_sample and len(l9)==num_sample:
                    break
            examples = l0+l1+l2+l3+l4+l5+l6+l7+l8+l9

        return examples

   
    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    


amazon_processors = {
    "yahoo": YahooProcessor,
}

amazon_output_modes = {
    "yahoo": "classification",
}

amazon_tasks_num_labels = {
    "yahoo": 10,
}