#pylint: disable=no-member

import pickle
import torch
from .model import ToxicCommentModel
from . import utils

class NNTextGen:
    def __init__(self):        
        vocab_path = utils.get_data_path("vocab.pkl")
        with open(vocab_path, "rb") as fin:
            self.posts_vocab, self.comments_vocab = pickle.load(fin)
        
        model_path = utils.get_data_path("toxic-model-max-40-100-0527.pt")
        model = ToxicCommentModel(self.posts_vocab, self.comments_vocab, torch.device('cpu'))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        self.model = model
    
    def predict(self, intext):
        ret = []
        for _ in range(10):
            comment = utils.predict(self.model, intext, 
                        self.posts_vocab, 
                        self.comments_vocab, torch.device('cpu'))
            ret.append("".join(comment))
        return ret
        