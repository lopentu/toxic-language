import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .preproc import convert_comment, convert_text

class ToxicDataset(Dataset):
    def __init__(self, src_df, tgt_df, src_vocab, tgt_vocab):
        self.data = []
        if src_df is not None and tgt_df is not None:
            self.build_data(src_df, tgt_df, src_vocab, tgt_vocab)

    def __repr__(self):
        return f"<ToxicDataset: {len(self)} samples>"

    def build_data(self, src_df, tgt_df, src_vocab, tgt_vocab):        
        for ridx, row in tgt_df.iterrows():
            try:
                txt = src_df.loc[row.TextId, :]
            except KeyError:
                print(f"Cannot find {row.TextId}")
                continue
            text_vec = convert_text(txt.TextContent, src_vocab)
            tgt_vec = convert_comment(row.CommentContent, tgt_vocab)
            self.data.append((text_vec, tgt_vec))            
        
        self.data = sorted(self.data, key=lambda x: len(x[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #pylint: disable=no-member
        src, tgt = self.data[idx]
        src_tensor = torch.LongTensor(src)
        tgt_tensor = torch.LongTensor(tgt)

        return (src_tensor, tgt_tensor)
    
    def save(self, fpath):
        with open(fpath, "wb") as fout:
            pickle.dump(self.data, fout)
    
    @staticmethod
    def load(fpath):
        with open(fpath, "rb") as fin:
            data = pickle.load(fin)
        inst = ToxicDataset(None, None, None, None)
        inst.data = data
        return inst

def collate_fn(batch_data, src_pad, tgt_pad):
    #pylint: disable=no-member
    src_data, tgt_data = zip(*batch_data)
    src_batch = pad_sequence(src_data, batch_first=True, padding_value=src_pad)
    tgt_batch = pad_sequence(tgt_data, batch_first=True, padding_value=tgt_pad)
    return (src_batch, tgt_batch)