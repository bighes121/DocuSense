"""BiLSTM-CRF implémenté de zéro (aucun poids pré-entraîné).

Représentation d'un mot = embedding du mot (minuscule) + CNN sur les caractères + indicateur de casse.
La couche CRF (algorithme forward pour la log-vraisemblance, Viterbi pour le décodage) est codée à la main.
"""
import torch
import torch.nn as nn

PAD, UNK = 0, 1


def case_feature(w):
    if w.isdigit(): return 1
    if any(c.isdigit() for c in w): return 2
    if w.isupper() and len(w) > 1: return 3
    if w[:1].isupper(): return 4
    if not w.isalnum(): return 5
    return 6


class Vocab:
    def __init__(self, sentences, min_freq=1):
        from collections import Counter
        wc = Counter(t.lower() for toks, _ in sentences for t, _, _ in toks)
        cc = Counter(c for toks, _ in sentences for t, _, _ in toks for c in t)
        self.words = ["<pad>", "<unk>"] + [w for w, n in wc.items() if n >= min_freq]
        self.chars = ["<pad>", "<unk>"] + sorted(cc)
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.c2i = {c: i for i, c in enumerate(self.chars)}

    def state(self):
        return {"words": self.words, "chars": self.chars}

    @classmethod
    def from_state(cls, st):
        v = cls.__new__(cls)
        v.words, v.chars = st["words"], st["chars"]
        v.w2i = {w: i for i, w in enumerate(v.words)}
        v.c2i = {c: i for i, c in enumerate(v.chars)}
        return v

    def encode(self, words, word_dropout=0.0):
        import random
        wid = [self.w2i.get(w.lower(), UNK) for w in words]
        if word_dropout:
            wid = [UNK if random.random() < word_dropout else i for i in wid]
        cid = [[self.c2i.get(c, UNK) for c in w[:20]] for w in words]
        return wid, cid, [case_feature(w) for w in words]


def collate(batch, vocab, word_dropout=0.0, with_labels=True):
    """batch: [(mots, ids_étiquettes)] -> tenseurs remplis (padding)."""
    enc = [vocab.encode(w, word_dropout) for w, _ in batch]
    B, T = len(batch), max(len(w) for w, _ in batch)
    L = max(len(c) for _, cs, _ in enc for c in cs)
    W = torch.zeros(B, T, dtype=torch.long)
    C = torch.zeros(B, T, L, dtype=torch.long)
    K = torch.zeros(B, T, dtype=torch.long)
    Y = torch.zeros(B, T, dtype=torch.long)
    M = torch.zeros(B, T, dtype=torch.bool)
    for b, ((wid, cid, case), (_, y)) in enumerate(zip(enc, batch)):
        n = len(wid)
        W[b, :n], K[b, :n], M[b, :n] = torch.tensor(wid), torch.tensor(case), True
        for t, cs in enumerate(cid):
            C[b, t, :len(cs)] = torch.tensor(cs)
        if with_labels:
            Y[b, :n] = torch.tensor(y)
    return W, C, K, Y, M


class CRF(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.trans = nn.Parameter(torch.randn(n_tags, n_tags) * 0.01)   # trans[i, j] = score i -> j
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def _score(self, em, y, mask):
        B, T, _ = em.shape
        s = self.start[y[:, 0]] + em[:, 0].gather(1, y[:, :1]).squeeze(1)
        for t in range(1, T):
            step = self.trans[y[:, t - 1], y[:, t]] + em[:, t].gather(1, y[:, t:t + 1]).squeeze(1)
            s = s + step * mask[:, t]
        last = y.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return s + self.end[last]

    def _log_partition(self, em, mask):
        alpha = self.start + em[:, 0]                                 # (B, n_tags)
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def nll(self, em, y, mask):
        return (self._log_partition(em, mask) - self._score(em, y, mask)).mean()

    def viterbi(self, em, mask):
        B, T, _ = em.shape
        score, back = self.start + em[:, 0], []
        for t in range(1, T):
            best, idx = (score.unsqueeze(2) + self.trans).max(dim=1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), nxt, score)
            back.append(idx)
        score = score + self.end
        paths = []
        for b in range(B):
            n = int(mask[b].sum())
            tag = int(score[b].argmax())
            path = [tag]
            for t in range(n - 2, -1, -1):
                tag = int(back[t][b, tag])
                path.append(tag)
            paths.append(path[::-1])
        return paths


class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_tags, wdim=100, cdim=30, cfilters=50, hidden=128, dropout=0.5):
        super().__init__()
        self.wemb = nn.Embedding(n_words, wdim, padding_idx=PAD)
        self.cemb = nn.Embedding(n_chars, cdim, padding_idx=PAD)
        self.cnn = nn.Conv1d(cdim, cfilters, kernel_size=3, padding=1)
        self.case = nn.Embedding(7, 8)
        self.lstm = nn.LSTM(wdim + cfilters + 8, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags)

    def emissions(self, W, C, K):
        B, T, L = C.shape
        ch = self.cemb(C.view(B * T, L)).transpose(1, 2)               # (B*T, cdim, L)
        ch = torch.relu(self.cnn(ch)).max(dim=2).values.view(B, T, -1)
        x = self.drop(torch.cat([self.wemb(W), ch, self.case(K)], dim=-1))
        h, _ = self.lstm(x)
        return self.out(self.drop(h))

    def loss(self, W, C, K, Y, M):
        return self.crf.nll(self.emissions(W, C, K), Y, M)

    def predict(self, W, C, K, M):
        return self.crf.viterbi(self.emissions(W, C, K), M)
