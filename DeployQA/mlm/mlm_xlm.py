""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""
import argparse
import glob
import logging
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
	          RobertaForMaskedLM,
                          AlbertConfig,
                          AlbertTokenizer,
	          AlbertForMaskedLM,                       
                          )

from utils import (compute_metrics,
                   output_modes, processors)

from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer),
}


class MyDataset(Dataset):

    def __init__(self, input_ids, special_tokens_mask, attention_mask):
        self.input_ids = input_ids
        self.special_tokens_mask = special_tokens_mask
        self.attention_mask = attention_mask

    def __getitem__(self, index):
        # return self.input_ids[index], self.special_tokens_mask[index], self.attention_mask[index]
        return {"input_ids": self.input_ids[index], "special_tokens_mask": self.special_tokens_mask[index],
                "attention_mask": self.attention_mask[index]}

    def __len__(self):
        return self.input_ids.size(0)


class PTInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, special_tokens_mask, attention_mask):
        self.input_ids = input_ids
        self.special_tokens_mask = special_tokens_mask
        self.attention_mask = attention_mask


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def _truncate_seq_pair(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()


def load_and_cache_examples(args, task, tokenizer, ttype='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    except:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)

        features = convert_pre_train_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                                          output_mode,
                                                          cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                          # xlnet has a cls token at the end
                                                          cls_token=tokenizer.cls_token,
                                                          sep_token=tokenizer.sep_token,
                                                          pad_token=1,
                                                          cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                          pad_on_left=bool(args.model_type in ['xlnet']),
                                                          # pad on the left for xlnet
                                                          pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_special_tokens_mask = torch.tensor([f.special_tokens_mask for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    print(all_input_ids[1])
    print(all_special_tokens_mask[1])
    print(all_attention_mask[1])

    # return {"input_ids": all_input_ids, "special_tokens_mask": all_special_tokens_mask, "attention_mask": all_attention_mask}
    return MyDataset(all_input_ids, all_special_tokens_mask, all_attention_mask)
    # return {"input_ids": all_input_ids}


def load_and_cache_examples_features(args, task, tokenizer, ttype='train', data_dir=""):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, args.test_file)
    except:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, args.test_file)

        features = convert_pre_train_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                                          output_mode,
                                                          cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                          # xlnet has a cls token at the end
                                                          cls_token=tokenizer.cls_token,
                                                          sep_token=tokenizer.sep_token,
                                                          pad_token=1,
                                                          cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                          pad_on_left=bool(args.model_type in ['xlnet']),
                                                          # pad on the left for xlnet
                                                          pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    return features


def get_train_dataset(args, task, tokenizer, langs_list,type):
    features = []
    all_input_ids = []
    all_special_tokens_mask = []
    all_attention_mask =[]
    if type == "train": datadir = os.path.join(args.data_dir, args.train_file)
    else: datadir = os.path.join(args.data_dir, args.dev_file)
    max_seq_length = args.max_seq_length
    tot = 0
    with open(datadir, "r", encoding='utf-8') as file:
        for line in file.readlines():
            if tot % 10000 == 0:
                print(line)
                print("read {}".format(tot))
            if tot == 4000000:
                break
            tot = tot +1
            cls_token=tokenizer.cls_token
            sep_token=tokenizer.sep_token
            tokens_a = tokenizer.tokenize(line)
            tokens_a = [cls_token] + tokens_a
            _truncate_seq_pair(tokens_a,max_seq_length-1)
            tokens_a = tokens_a + [sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
            padding_length = max_seq_length - len(input_ids)
            attention_mask = ([1] * len(input_ids)) + ([0] * padding_length) 
            specialtokens_mask =  ([1] + [0] * (len(input_ids)-2) +[1]) + ([1] * padding_length)
            input_ids = input_ids + ([1] * padding_length)
            #print(inputs)
            #input_ids = inputs["input_ids"]
            #attention_mask = inputs["attention_mask"]
            #special_tokens_mask = []
            #for att in attention_mask:
            #specialtokens_mask = [1 - x for x in attention_mask]
            #special_tokens_mask.append(specialtokens_mask)
            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(specialtokens_mask) == max_seq_length
            all_input_ids.append(input_ids)
            all_special_tokens_mask.append(specialtokens_mask)
            all_attention_mask.append(attention_mask)

            
    print(all_input_ids[1])
    print(all_special_tokens_mask[1])
    print(all_attention_mask[1])

    return MyDataset(torch.tensor(all_input_ids,dtype=torch.long), torch.tensor(all_special_tokens_mask,dtype=torch.long), torch.tensor(all_attention_mask,dtype=torch.long))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='codesearch', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--train_file", default="train_top10_concat.tsv", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="test file")
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training

    # device_ids = [0, 1, 2]
    # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    model.to(args.device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
    langs_list = {"NewPL_2"}
    train_dataset = get_train_dataset(args, args.task_name, tokenizer, langs_list,"train")
    eval_dataset = get_train_dataset(args, args.task_name, tokenizer, langs_list,"eval")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy = "steps",
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        learning_rate=args.learning_rate,
        #lr_scheduler_type="constant_with_warmup",
        warmup_steps=1000,
        logging_dir="./models",
        logging_steps=args.logging_steps,
        save_steps=20000,
        save_total_limit=5,
        eval_steps = 1000
    )
    logger.info(training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
