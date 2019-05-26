from pathlib import Path

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