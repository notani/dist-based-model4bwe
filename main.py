#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from tqdm import tqdm
import argparse
import gzip
import lzma
import numpy as np
import time

from torch import nn
from torch import optim
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from common import init_logger

verbose = False
logger = None
use_cuda = False


class CBOW(nn.Module):
    """Continuous bag-of-words model."""
    def __init__(self, vocab_size, dim_embed=50, *args, **kwargs):
        "Initialize a network."
        super(CBOW, self).__init__()
        self.verbose = kwargs.get('verbose', False)
        self.logger = init_logger('CBOW')
        self.embeddings = nn.Embedding(vocab_size, dim_embed, sparse=True)
        self.l1 = nn.Linear(dim_embed, vocab_size)

    def forward(self, X):
        context = self.embeddings(X).mean(dim=1)
        return self.l1(context)

    def get_embeddings(self):
        """Return embeddings."""
        dat = self.embeddings.weight.data
        if use_cuda:
            dat = dat.cpu()
        return dat.numpy()


class Corpus():
    """Corpus reader."""
    def __init__(self, window_size, *args, **kwargs):
        "Set variables."
        self.verbose = kwargs.get('verbose', False)
        self.logger = init_logger('Corpus')
        self.window_size = window_size  # length of window on each side
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.UNK = self.w2i['<unk>']
        self.BOS = self.w2i['<s>']
        self.EOS = self.w2i['</s>']

    def set_i2w(self):
        """Initialize an i2w indexer."""
        self.i2w = [w for w, _ in sorted(self.w2i.items(), key=lambda t: t[1])]

    def read(self, path_corpus):
        """Read a corpus and convert it into a matrix of word indices."""
        if path_corpus.endswith('.xz'):
            f = lzma.open(path_corpus, 'rt')
        elif path_corpus.endswith('.gz'):
            f = gzip.open(path_corpus, 'rt')
        else:
            f = open(path_corpus, 'rt')
        if self.verbose:
            self.logger.info('Read from ' + path_corpus)
        for line in f:
            words = line.strip().split()
            if len(words) < 2 * self.window_size - 1:
                continue
            indices = [self.BOS] + [self.w2i[w] for w in words] \
                      + [self.EOS]
            yield indices
        if self.verbose:
            self.logger.info('Done.')
        f.close()

    def get_vocabsize(self):
        return len(self.w2i)



def generate_batch(sents, window, batch_size):
    """Generate batches."""
    contexts, targets, count = [], [], 0
    count = 0
    for sent in sents:
        l = len(sent)
        for pos in range(window, l - window):
            context = sent[pos - window:pos]
            context += sent[pos + 1:pos + window + 1]
            contexts.append(context)
            targets.append(sent[pos])
            count += 1
            if count >= batch_size:
                contexts = Variable(torch.LongTensor(contexts))
                targets = Variable(torch.LongTensor(targets))
                if use_cuda:
                    contexts = contexts.cuda()
                    targets = targets.cuda()
                yield contexts, targets
                contexts, targets, count = [], [], 0


def save_embeddings(filename, embs, i2w):
    if verbose:
        logger.info('Save embeddings to ' + filename)
    with open(filename, 'w') as f:
        f.write('{} {}\n'.format(*embs.shape))
        for w, emb in zip(i2w, embs):
            f.write('{} {}'.format(w, ' '.join(str(v) for v in emb)))


def main(args):
    global verbose
    verbose = args.verbose

    global use_cuda
    use_cuda = args.cuda

    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    lr = args.lr
    batch_size = args.batch_size
    window_size = args.window_size
    corpus = Corpus(window_size=window_size, verbose=verbose)
    train = list(corpus.read(args.path_corpus))
    corpus.set_i2w()

    model = CBOW(corpus.get_vocabsize(), verbose=verbose)
    if use_cuda:
        model.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for ITER in range(1):
        np.random.shuffle(train)
        train_words, train_loss = 0, 0.0
        start = time.time()
        for contexts, targets in tqdm(generate_batch(train, window_size, batch_size)):
            model.zero_grad()
            loss = loss_func(model(contexts), targets)
            loss.backward()
            optimizer.step()
            train_words += batch_size
            train_loss += float(loss.data)
            break
        print('[{}] loss/word = {:.4f}, ppl={:.4f}, time = {:.2f}'.format(
            ITER+1, train_loss / train_words, np.exp(train_loss / train_words),
            time.time() - start))

    # Save vectors
    embs = model.get_embeddings()
    save_embeddings(args.path_output, embs, corpus.i2w)
    return 0


if __name__ == '__main__':
    logger = init_logger('MAIN')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_corpus', help='path to a corpus file')
    parser.add_argument('--window-size', type=int, default=2,
                        help='window size on each side')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--seed', dest='random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True, help='path to an output file')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)