#pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

class ToxicCommentModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, device):
        super(ToxicCommentModel, self).__init__()
        self.device = device
        embeddim = src_vocab.vectors.shape[1]
        nvocab = len(src_vocab.stoi)
        n_hidden = 40
        tgt_embed_dim = 100
        self.max_out_len = 10
        self.tgt_soi = tgt_vocab.stoi["<SOS>"]
        n_tgt = len(tgt_vocab.stoi)

        self.embed = nn.Embedding(nvocab, embeddim, padding_idx=src_vocab.stoi["<PAD>"])
        self.embed.weight = nn.Parameter(torch.FloatTensor(src_vocab.vectors))
        self.embed.weight.requires_grad = False
        self.tgt_embed = nn.Embedding(n_tgt, tgt_embed_dim, padding_idx=tgt_vocab.stoi["<PAD>"])
        self.tgt_embed.weight.requires_grad = True

        self.avg = nn.AdaptiveMaxPool2d((1, embeddim//2))
        self.fc1 = nn.Linear(embeddim//2, n_hidden)
        self.rnn = nn.GRU(tgt_embed_dim, n_hidden, batch_first=True)
        self.fc2 = nn.Sequential(nn.Linear(n_hidden, n_tgt), nn.Tanh())

    def forward(self, X, Y=None):
        # encoding
        embed_x = self.embed(X).unsqueeze(1)
        avg_x = self.avg(embed_x).squeeze()
        hidden = torch.tanh(self.fc1(avg_x))

        # decoding
        rnn_hidden = hidden.unsqueeze(0)
        n_batch = X.shape[0]
        max_seq_len = Y.shape[1] if Y is not None else self.max_out_len
        tgt_embed_dim = self.tgt_embed.embedding_dim
        n_tgt = self.tgt_embed.num_embeddings

        # set first SOS token
        soi_tensor = torch.LongTensor([self.tgt_soi] * n_batch).to(self.device)
        rnn_input = self.tgt_embed(soi_tensor).unsqueeze(1)        
        rnn_input = F.relu(rnn_input)

        # output preallocation
        outputs = torch.zeros((n_batch, max_seq_len, n_tgt)).to(self.device)

        if Y is not None:
            # use teacher forcing
            for yi in range(Y.shape[1]):

                rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
                tgt_out = self.fc2(rnn_out.squeeze())             
                outputs[:, yi, :] = F.log_softmax(tgt_out, dim=1)

                # set current target as next input
                yvec = Y[:, yi]
                rnn_input = self.tgt_embed(yvec).unsqueeze(1)
                rnn_input = F.relu(rnn_input)

        else:

            for yi in range(max_seq_len):                
                rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
                tgt_out = self.fc2(rnn_out.squeeze())
                rnn_input = self.tgt_embed(tgt_out.argmax(1)).unsqueeze(1)
                rnn_input = F.relu(rnn_input)
                outputs[:, yi, :] = F.log_softmax(tgt_out, dim=1)

        return outputs
    
