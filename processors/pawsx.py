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


class PawsxProcessor(DataProcessor):
    """Processor for the pawsx dataset."""
    
    def __init__(self):
        self.dataset_name = 'paws-x'
    
    def get_test_dataset(self, args):
        dataset = load_dataset(self.dataset_name, 'en', split='test')
        test_set = [text_a+'\ '+text_b for text_a, text_b in zip(dataset['sentence1'], dataset['sentence2'])]
        labels = dataset['label']
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
        for i, data_ex in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['sentence1']
            text_b = data_ex['sentence2']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2 = [], [], []
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

    def get_train_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='train', num_sample=num_sample)

    def get_dev_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='dev', num_sample=num_sample)

    def get_test_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='test', num_sample=num_sample)

    def get_translate_train_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        examples = []
        for lg in language.split(','):
            file_path = os.path.join(data_dir, "XNLI-Translated/en-{}-translated.tsv".format(lg))
            logger.info("reading file from " + file_path)
            lines = self._read_tsv(file_path)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % ("translate-train", lg, i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        if num_sample != -1:
            examples = random.sample(examples, num_sample)
        return examples

    def get_translate_test_examples(self, data_dir, language='en', num_sample=-1):
        lg = language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-Translated/test-{}-en-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_pseudo_test_examples(self, data_dir, language='en', num_sample=-1):
        lines = self._read_tsv(
            os.path.join(data_dir, "XNLI-Translated/pseudo-test-set/en-{}-pseudo-translated.csv".format(language)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("pseudo-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1]


class PawsxMultiProcessor(DataProcessor):
    """Processor for the pawsx multilingual dataset."""
    
    def __init__(self):
        self.dataset_name = 'paws-x'
    
    def get_test_dataset(self, args):
        dataset = load_dataset(self.dataset_name, 'en', split='test')
        test_set = [text_a+'\ '+text_b for text_a, text_b in zip(dataset['sentence1'], dataset['sentence2'])]
        labels = dataset['label']
        return test_set, labels

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
        for i, data_ex in enumerate(dataset):
            guid = str(i)
            text_a = data_ex['sentence1']
            text_b = data_ex['sentence2']
            label = data_ex['label']
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, int)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2 = [], [], []
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

    def get_train_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='train', num_sample=num_sample)

    def get_dev_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='dev', num_sample=num_sample)

    def get_test_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='test', num_sample=num_sample)

    def get_translate_train_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        examples = []
        for lg in language.split(','):
            file_path = os.path.join(data_dir, "XNLI-Translated/en-{}-translated.tsv".format(lg))
            logger.info("reading file from " + file_path)
            lines = self._read_tsv(file_path)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % ("translate-train", lg, i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        if num_sample != -1:
            examples = random.sample(examples, num_sample)
        return examples

    def get_translate_test_examples(self, data_dir, language='en', num_sample=-1):
        lg = language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-Translated/test-{}-en-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_pseudo_test_examples(self, data_dir, language='en', num_sample=-1):
        lines = self._read_tsv(
            os.path.join(data_dir, "XNLI-Translated/pseudo-test-set/en-{}-pseudo-translated.csv".format(language)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("pseudo-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1]


xnli_processors = {
    "pawsx": PawsxProcessor,
}

xnli_output_modes = {
    "pawsx": "classification",
}

xnli_tasks_num_labels = {
    "pawsx": 2,
}