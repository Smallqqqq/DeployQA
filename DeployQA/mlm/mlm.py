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

tokenizer = RobertaTokenizerFast.from_pretrained("../added_vocab_models")
model = RobertaForMaskedLM.from_pretrained("../added_vocab_models")
print(len(tokenizer))
model.to(device)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="output.txt",
    block_size=256,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print("finish datacollator")

training_args = TrainingArguments(
        output_dir="../mlm_models",
        overwrite_output_dir=True,
        per_device_train_batch_size=12,
        learning_rate=1e-4,
        gradient_accumulation_steps=50,
        warmup_steps=10000,
        logging_dir="../trainlog_mlm",
        logging_steps=10,
        save_strategy="steps",
        save_steps=2500,
        save_total_limit=50,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        adam_beta2=0.99,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("../mlm_models")