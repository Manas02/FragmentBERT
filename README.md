# Masked Language Models are Fragment Based Drug Designers

#### Step zero
## Unzip the `train.txt.zip`

Install Dependencies :

```
pip3 install datasets transformers
```

```
python3 pretrain.py
```

```
python3 SMILESBERT.py
```

To visualize

```
tensorboard --logdir=./smiles-bert/runs
```
and then go to http://localhost:6006/

---

To Do:

- [x] Create a Perfect `Tokenizer`
- [ ] Create a Perfect `Fragment Builder`
- [x] Train and Check the Progress of current model
