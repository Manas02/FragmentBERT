# Masked Language Models are Fragment Based Drug Designers

#### Step zero

```sh
unzip train.txt.zip
```

Install Dependencies :

```
pip3 install transformers
```

```
python3 pretrain.py
```

```
python3 model.py
```

To visualize

```
tensorboard --logdir=./smiles-bert/runs
```
and then go to http://localhost:6006/

```
python3 predict.py
```
---

To Do:

- [ ] Create a Perfect `Fragment Builder`
