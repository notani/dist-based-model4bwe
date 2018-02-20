# dist-based-model4bwe
PyTorch implementation of "A Distribution-based Model to Learn Bilingual Word Embeddings" (Cao et al., COLING2016)


```shell
conda install pytorch -c pytorch -y
conda install tqdm
```

```shell
python main.py --src en:data/ukWaC/tokenized.mini.txt.xz --trg it:data/itWaC/tokenized.mini.txt.xz -o vectors.txt --batch-size 1024 -v
```

```
2018-02-20 09:12:02,417/Corpus[INFO]: Read from data/ukWaC/tokenized.mini.txt.xz
2018-02-20 09:12:02,635/Corpus[INFO]: Done.
2018-02-20 09:12:02,636/Corpus[INFO]: Read from data/itWaC/tokenized.mini.txt.xz
2018-02-20 09:12:02,888/Corpus[INFO]: Done.
256it [00:44,  5.72it/s]
[1] loss = -7821.9537 (-8056.8559/234.9022), time = 44.74
256it [00:40,  6.30it/s]
[2] loss = -7914.2870 (-8110.2862/195.9992), time = 40.61
256it [00:42,  5.97it/s]
[3] loss = -7972.2588 (-8168.2426/195.9838), time = 42.92
256it [00:50,  5.11it/s]
[4] loss = -8028.0803 (-8224.0569/195.9766), time = 50.11
256it [00:58,  4.40it/s]
[5] loss = -8083.8612 (-8279.8418/195.9806), time = 58.12
```

3x faster if you use GPU.


```python
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
model.most_similar('en:cat')
> [('en:soil', 0.5583387613296509), ('en:consequently', 0.5522608160972595), ('it:maialoooooo', 0.5325822830200195), ('en:coding', 0.5320591926574707), ('it:curricolari', 0.5157815217971802), ('it:efficaci', 0.5052497386932373), ('en:impact', 0.5048317909240723), ('en:psb', 0.49141693115234375), ('en:cheque', 0.48973801732063293), ('en:shiver', 0.4879668056964874)]
```

Umm...
