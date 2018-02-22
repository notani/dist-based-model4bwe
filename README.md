# dist-based-model4bwe
PyTorch implementation of "A Distribution-based Model to Learn Bilingual Word Embeddings" (Cao et al., COLING2016)


```shell
conda install pytorch -c pytorch -y
conda install tqdm
```

```shell
python main.py --src en:data/ukWaC/tokenized.mini.txt.xz --trg it:data/itWaC/tokenized.mini.txt.xz -o vectors.txt --batch-size 1024 --cuda -v
# CPU: python main.py --src en:data/ukWaC/tokenized.mini.txt.xz --trg it:data/itWaC/tokenized.mini.txt.xz -o vectors.txt --batch-size 1024 -v
```

```
2018-02-20 10:19:02,241/Corpus[INFO]: Read from data/ukWaC/tokenized.mini.txt.xz
2018-02-20 10:19:02,452/Corpus[INFO]: Done.
2018-02-20 10:19:02,453/Corpus[INFO]: Read from data/itWaC/tokenized.mini.txt.xz
2018-02-20 10:19:02,700/Corpus[INFO]: Done.
2018-02-20 10:19:12,779/MAIN[INFO]: window size: 2
2018-02-20 10:19:12,779/MAIN[INFO]: learning rate: 0.01
2018-02-20 10:19:12,779/MAIN[INFO]: batch size: 1024
256it [00:17, 14.96it/s]
[1] loss = 8331.1028 (8096.2456/234.8572), time = 17.11
2018-02-20 10:19:30,112/MAIN[INFO]: Save embeddings to vectors.txt
256it [00:16, 15.26it/s]
[2] loss = 8242.8091 (8047.9243/194.8848), time = 16.78
2018-02-20 10:19:49,061/MAIN[INFO]: Save embeddings to vectors.txt
256it [00:16, 15.23it/s]
[3] loss = 8174.0800 (7979.4080/194.6720), time = 16.82
2018-02-20 10:20:07,977/MAIN[INFO]: Save embeddings to vectors.txt
256it [00:16, 15.34it/s]
[4] loss = 8144.2930 (7949.8464/194.4466), time = 16.69
2018-02-20 10:20:26,840/MAIN[INFO]: Save embeddings to vectors.txt
256it [00:16, 15.46it/s]
[5] loss = 8101.0505 (7906.8492/194.2012), time = 16.56
2018-02-20 10:20:45,467/MAIN[INFO]: Save embeddings to vectors.txt
```

```python
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
model.most_similar('en:cat')
> [('en:folk-tales', 0.5345577597618103), ('en:inject', 0.5104585886001587), ('it:mariotti', 0.4928727149963379), ('it:ebbero', 0.48998019099235535), ('it:funzionano', 0.4896196126937866), ('it:trash', 0.47888505458831787), ('en:feudal', 0.47887367010116577), ('en:creeps', 0.47296014428138733), ('it:abbronzati', 0.4703264832496643), ('en:staging', 0.4651801586151123)]
```

# Cross-ligual learning

```shell
python main.py --src en:data/ukWaC/tokenized.1m.txt.xz --trg it:data/itWaC/tokenized.1m.txt.xz -o vectors.en.it.1m.d128.mf10.txt --batch-size 64 --lr 0.5 --model models/en.it/full.1m --iter 100 --min-freq 10 --dim 128 -v
```
