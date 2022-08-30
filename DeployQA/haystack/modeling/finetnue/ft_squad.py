from haystack.reader.farm import FARMReader
from pathlib import Path
import torch
import os


def finetune_model(data_dir,train_filename,dev_filename,save_dir,model_name,my_saved_model_name,batchs,epochs,learning,noans,noans_boost):

    reader = FARMReader(model_name_or_path=model_name,
                        use_gpu=True,
                        no_ans_boost=noans_boost,
                        return_no_answer=noans,
                        top_k_per_candidate=3,
                        top_k_per_sample=1,
                        max_seq_len=256,
                        doc_stride=128)

    reader.train(data_dir=data_dir,
                 train_filename=train_filename,
                 dev_filename=dev_filename,
                 use_gpu=True,
                 batch_size=batchs ,
                 evaluate_every=300,
                 n_epochs=epochs,
                 save_dir=save_dir,
                 learning_rate=learning,
                 logpath=save_dir,
                 checkpoint_every=3400,
                 checkpoint_root_dir=Path(save_dir),
                 logevery=10)
    reader.save(save_dir)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

    finetune_model(data_dir = "../squad",
                   train_filename='train-v2.0.json',
                   dev_filename='DeQuAD_test.json',
                   my_saved_model_name="test_output",
                   save_dir = "../finetuned_on_squad",
                   model_name = "roberta-squad",
                   batchs = 48,
                   epochs = 4,
                   learning = 3e-5,
                   noans=True,
                   noans_boost=0
    )
