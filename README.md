# dist-based-model4bwe
PyTorch implementation of "A Distribution-based Model to Learn Bilingual Word Embeddings" (Cao et al., COLING2016)


```shell
conda install pytorch -c pytorch -y
conda install tqdm
```

```shell
python main.py --src en:data/ukWaC/tokenized.mini.txt.xz --trg it:data/itWaC/tokenized.mini.txt.xz -o vectors.txt --batch-size 1 --lr 0.001 -v
```
