from torch.utils.data import Dataset
import torch
from .preproc import convert_comment, convert_text

class ToxicDataset(Dataset):
    def __init__(self, src_df, tgt_df, src_vocab, tgt_vocab):
        self.src = []
        self.tgt = []
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
            self.src.append(text_vec)
            self.tgt.append(tgt_vec)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        #pylint: disable=no-member
        src_tensor = torch.LongTensor(self.src[idx]).unsqueeze(0)
        tgt_tensor = torch.LongTensor(self.tgt[idx]).unsqueeze(0)

        return (src_tensor, tgt_tensor)