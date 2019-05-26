import jieba

def convert_text(text, vocab):
    words_iter = jieba.cut(text)
    num_vec = vocab.to_numerics(words_iter)
    return num_vec

def convert_comment(text, vocab):
    sos_tok = [vocab.stoi["<SOS>"]]
    comments_tok = vocab.to_numerics(text.split("\\"))
    eos_tok = [vocab.stoi["<EOS>"]]
    return sos_tok + comments_tok + eos_tok

def inverse_convert_comment(comm_vec, vocab):
    return vocab.to_texts(comm_vec)