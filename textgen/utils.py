from pathlib import Path
import numpy as np
import torch
from .preproc import convert_text, inverse_convert_comment

def get_data_path(filename):
    fpath = Path(__file__).parent / f"data/{filename}"
    return fpath

def get_resource_path(filename):
    fpath = Path(__file__).parent / f"resources/{filename}"
    return fpath

def get_seq_overlap(ypred, y, vocab):        
    eos_i = vocab.stoi["<EOS>"]
    eos_idx = y.tolist().index(eos_i)
    ans_set = set(y[:eos_idx])
    pred_set = set(ypred[:eos_idx])    
        
    return len(ans_set & pred_set) / len(ans_set)

def predict(model, intext, src_vocab, tgt_vocab, device):
    try:
        text_vec = np.vstack([convert_text(intext, src_vocab)] * 2)
        in_tensor = torch.LongTensor(text_vec).to(device)   # pylint: disable=no-member
        out = model(in_tensor)    
        topv, topi = out.topk(10)
        topv = topv[0].cpu().detach().numpy()
        topi = topi[0].cpu().detach().numpy()
        comments = decode_seq(topv, topi, tgt_vocab)
    except Exception as ex:
        print(ex)
        comments = "...."
    return comments

def decode_seq(topv, topi, vocab):    

    visited = set()
    rs = np.random.RandomState()    #pylint: disable=no-member
    ret = []
    for step_i in range(topv.shape[1]):
        logits = topv[step_i]
        tok_idx = topi[step_i]
            
        for vec_idx, _ in enumerate(logits):        
            tok_i = tok_idx[vec_idx]
            if tok_i in visited or tok_i >= vocab.stoi["<SOS>"]:
                logits[vec_idx] *= 10
        
        scores = np.exp(logits).astype('float64')
        probs = scores/np.sum(scores)
        selected_idx = np.argwhere(rs.multinomial(1, probs)).flatten()[0]
        selected = tok_idx[selected_idx]    
        visited.add(selected)
        if rs.binomial(1, step_i * 0.1):
            break
        ret.append(vocab.to_texts([selected])[0])
    return ret