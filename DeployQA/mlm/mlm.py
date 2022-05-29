from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import RobertaConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import torch
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
print(device)
tokenizer = RobertaTokenizerFast.from_pretrained("./model/tokens")
#tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
config = RobertaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)
print(len(tokenizer))
model.to(device)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="output.txt",
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print("finish datacollator")
training_args = TrainingArguments(
        output_dir="mlm",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=12,
        learning_rate=1e-4,
        warmup_steps=1000,
        logging_dir="mlm",
        logging_steps=200,
        save_steps=20000,
        save_total_limit=5
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("mlm_dir")