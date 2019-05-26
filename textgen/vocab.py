from itertools import chain
import numpy as np

class Vocabulary:
    TOKENS = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]

    def __init__(self, itos={}, stoi={}, vectors=None):
        self.itos = itos
        self.stoi = stoi
        self.vectors = vectors

        if vectors:
            self.initialize_vectors()

    def __repr__(self):
        if isinstance(self.vectors, np.ndarray):
            return f"<Vocabulary: {len(self.itos)} items, emb dim: {self.vectors.shape}>"
        else:
            return f"<Vocabulary: {len(self.itos)} items>"

    def initialize_vectors(self):
        TOKENS = Vocabulary.TOKENS
        embdim = len(self.vectors[0])
        rs = np.random.RandomState(3234)    #pylint: disable=no-member
        for t in TOKENS:
            tidx = len(self.itos)
            self.stoi[t] = tidx
            self.itos[tidx] = t

            if t != "<PAD>":
                self.vectors.append(rs.rand(embdim))
            else:
                self.vectors.append(np.zeros(embdim))
        self.vectors = np.vstack(self.vectors)

    def build_vocabulary(self, tokens):
        TOKENS = Vocabulary.TOKENS
        stoi = self.stoi
        itos = self.itos

        for tok in chain.from_iterable([tokens, TOKENS]):
            if tok not in stoi:
                tidx = len(itos)
                stoi[tok] = tidx
                itos[tidx] = tok

        self.vectors = None
    
    def to_numerics(self, tokens):
        stoi = self.stoi
        return [stoi[x] for x in tokens if x in stoi]
    
    def to_texts(self, vectors):
        itos = self.itos
        return [itos[i] for i in vectors if i in itos]




