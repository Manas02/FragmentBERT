# Masked Language Models are Fragment Based Drug Designers
---
#### Step zero
## Unzip the `train.txt.zip`
---
Install Dependencies :

```
pip3 install datasets transformers
```

then 

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

- [ ] Create a Perfect `Tokenizer`
- [ ] Create a Perfect `Fragment Builder`
- [ ] Train and Check the Progress of current model
