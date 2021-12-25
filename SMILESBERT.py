#!/usr/bin/env python
# coding: utf-8
# Author : Manas Mahale <manas.mahale@bcp.edu.in>

from datasets import load_dataset
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline
from tokenizers import BertWordPieceTokenizer 
import os
import json

files = ["train.txt"]

dataset = load_dataset(".", data_files=files, split="train")


d = dataset.train_test_split(test_size=0.1)
d["train"], d["test"]


special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]


vocab_size = 1_000
max_length = 256
truncate_longer_samples = True


tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)


model_path = "smiles-bert"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

tokenizer.save_model(model_path)

with open(os.path.join(model_path, "config.json"), "w") as f:
    tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
      }
    json.dump(tokenizer_cfg, f)

# when the tokenizer is trained and configured, load it as BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)


def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)


# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset = d['train'].map(encode, batched=True)


# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)


if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as 
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
train_dataset, test_dataset


# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True, batch_size=2_000,
                                    desc=f"Grouping texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True, batch_size=2_000,
                                  num_proc=4, desc=f"Grouping texts in chunks of {max_length}")

len(test_dataset)

# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)


# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=100,           # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=10,               # evaluate, log and save model checkpoints every 10 step
    save_steps=10,
    load_best_model_at_end=True,    # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=2,             # whether you don't have much space so you let only 2 model weights saved in the disk
)

# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# train the model
trainer.train()

# when you load from pretrained
# model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000"))
# tokenizer = BertTokenizerFast.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# perform predictions
example = "CN(Cc1cccs1)C(=O)Cn1c(=O)c(C#N)c2n(c1=O)c[mask]CC2"
for prediction in fill_mask(example):
    print(prediction)
