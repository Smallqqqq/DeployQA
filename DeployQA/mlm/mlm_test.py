
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""
import argparse
import glob
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          AutoTokenizer,
                          RobertaTokenizer)

from utils import (compute_metrics,
                        output_modes, processors)

from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def main():
        tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base")
        inputs = tokenizer(
            "3541",
            "chaiyt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation='only_first',
            return_overflowing_tokens=True,
        )
        print(inputs)
        attention_mask = inputs["attention_mask"]
        special_tokens_mask = []
        for att in attention_mask:
            specialtokens_mask = [1-x for x in att]
            special_tokens_mask.append(specialtokens_mask)
        print(special_tokens_mask)
if __name__ == "__main__":
    main()
