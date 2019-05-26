#pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

class ToxicCommentModel(nn.Module):    
    def __init__(self, src_vocab, tgt_vocab):
        super(ToxicCommentModel, self).__init__()
        embeddim = src_vocab.vectors.shape[1]
        nvocab = len(src_vocab.stoi)
        n_feats = 20
        n_hidden = 20
        tgt_embed_dim = 20
        self.max_out_len = 20
        self.tgt_soi = tgt_vocab.stoi["<SOS>"]
        n_tgt = len(tgt_vocab.stoi)
        
        self.embed = nn.Embedding(nvocab, embeddim, padding_idx=src_vocab.stoi["<PAD>"])        
        self.embed.weight = nn.Parameter(torch.FloatTensor(src_vocab.vectors))
        self.embed.weight.requires_grad = False
        self.tgt_embed = nn.Embedding(n_tgt, tgt_embed_dim, padding_idx=tgt_vocab.stoi["<PAD>"])
        self.tgt_embed.weight.requires_grad = False
                
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_feats, kernel_size=(3, embeddim), stride=(3, embeddim)),
            nn.ReLU())
        self.fc1 = nn.Linear(n_feats, n_hidden)
        self.rnn = nn.GRU(tgt_embed_dim, n_hidden, batch_first=True)
        self.fc2 = nn.Linear(n_feats, n_tgt)        
    
    def forward(self, X, Y=None):
        # encoding
        embed_x = self.embed(X).unsqueeze(1)        
        conv1_x = self.conv1(embed_x)
        feats = torch.max_pool2d(conv1_x, kernel_size=conv1_x.shape[2:]).squeeze()
        hidden = self.fc1(feats)
        
        # decoding
        rnn_hidden = hidden.unsqueeze(0)
        n_batch = X.shape[0]
        max_seq_len = Y.shape[1] if Y is not None else self.max_out_len
        tgt_embed_dim = self.tgt_embed.embedding_dim
        n_tgt = self.tgt_embed.num_embeddings
        outputs = torch.zeros((n_batch, max_seq_len, n_tgt))
        if Y is not None:
            # use teacher forcing
            for yi in range(Y.shape[1]):
                yvec = Y[:, yi]
                rnn_input = self.tgt_embed(yvec).unsqueeze(1)
                rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
                tgt_out = self.fc2(rnn_out.squeeze())
                outputs[:, yi, :] = tgt_out
            
        else:
            soi_tensor = torch.LongTensor([self.tgt_soi] * n_batch)
            rnn_input = self.tgt_embed(soi_tensor).unsqueeze(1)
            for yi in range(max_seq_len):                                
                rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
                tgt_out = self.fc2(rnn_out.squeeze())                
                outputs[:, yi, :] = tgt_out                        
        
        return outputs