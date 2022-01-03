#!/usr/bin/env python
# coding: utf-8
# Author : Manas Mahale <manas.mahale@bcp.edu.in>
# Large Languge Models are Fragment Based Drug Designers

from os import mkdir
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import (PreTrainedTokenizerFast, 
                            LineByLineTextDataset, 
                            DataCollatorForLanguageModeling, 
                            BertConfig , 
                            BertForMaskedLM, 
                            TrainingArguments, 
                            Trainer)


files = ["smiles/canonical_train_scaffold.txt", "smiles/canonical_test.txt"]

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = WhitespaceSplit()

tokenizer.train(files, trainer)

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

max_length = 64
vocab_size = tokenizer.get_vocab_size()

model_path = './model/'
tokenizer_path = './tokenizer/'

mkdir(model_path)
mkdir(tokenizer_path)

tokenizer.enable_truncation(max_length=max_length)

tokenizer.save('tokenizer/tokenizer.json')

tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tokenizer.json")

tokenizer.mask_token = "[MASK]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"
tokenizer.sep_token = "[SEP]"
tokenizer.cls_token = "[CLS]"


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./smiles/canonical_train_scaffold.txt",
    block_size=128,
)

test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./smiles/canonical_test.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=250,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1,                # evaluate, log and save model checkpoints every x step
    save_steps=10,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=1,           # whether you don't have much space so you let only x model weights saved in the disk
)

# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# train the model
trainer.train()
