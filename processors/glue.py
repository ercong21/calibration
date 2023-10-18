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
""" cola utils (dataset loading and evaluation) """

import logging
import os
import random

from transformers import DataProcessor
from .utils import InputExample
from datasets import load_dataset

logger = logging.getLogger(__name__)


class ColaProcessor(DataProcessor):
    """Processor for the CoLA dataset."""
    
    def __init__(self):
        self.dataset_name = 'cola'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='validation')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text = data_ex['sentence']
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


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 dataset."""
    
    def __init__(self):
        self.dataset_name = 'sst2'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='validation')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text = data_ex['sentence']
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


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC dataset."""
    
    def __init__(self):
        self.dataset_name = 'mrpc'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='test')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['sentence1']
            text_b = data_ex['sentence2']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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
    

class QqpProcessor(DataProcessor):
    """Processor for the QQP dataset."""
    
    def __init__(self):
        self.dataset_name = 'qqp'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='test')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['question1']
            text_b = data_ex['question2']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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

    

class RteProcessor(DataProcessor):
    """Processor for the RTE dataset."""
    
    def __init__(self):
        self.dataset_name = 'rte'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='test')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['sentence1']
            text_b = data_ex['sentence2']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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
    

class QnliProcessor(DataProcessor):
    """Processor for the QNLI dataset."""
    
    def __init__(self):
        self.dataset_name = 'qnli'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='test')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['question']
            text_b = data_ex['sentence']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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
    

class WnliProcessor(DataProcessor):
    """Processor for the QNLI dataset."""
    
    def __init__(self):
        self.dataset_name = 'wnli'

    def get_test_dataset(self, args):
        dataset = load_dataset('glue', self.dataset_name, split='test')
        if args.num_sample != -1:
            dataset = dataset.select(random.sample(range(len(dataset)), args.num_sample))
        test_set = dataset['sentence']
        labels = dataset['label']
        
        return test_set, labels

    def get_examples(self, args, split):
        """See base class."""
        assert split in ['test', 'train'], "split does not exits"
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('glue', self.dataset_name, split=split)
        num_sample = args.num_train_sample if split=='train' else args.num_test_sample
        examples = []

        for (i, data_ex) in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['sentence1']
            text_b = data_ex['sentence2']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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


glue_processors = {
    "cola": ColaProcessor,
    'sst2': Sst2Processor,
}

glue_output_modes = {
    "cola": "classification",
    "sst2": "classification",
}

glue_tasks_num_labels = {
    "cola": 2,
    "sst2": 2,
}