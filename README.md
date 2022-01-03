# Masked Language Models are Fragment Based Drug Designers

Install Dependencies :

```
pip3 install transformers
```

```
python3 src/pretrain.py
```

```
python3 src/model.py
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

- [ ] **Join molecules**
- [ ] Consistent data file path
- [ ] Document
- [ ] Web app
- [ ] `pip install fbdd`
