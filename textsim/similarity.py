#pylint: disable=no-member
import pickle
import numpy as np
import jieba
from .utils import get_data_path

class SimilarComments:

    def __init__(self):
        svd_path = get_data_path("svd.pkl")
        tfidf_path = get_data_path("tfidf.pkl")
        materials_path = get_data_path("text_materials.pkl")

        self.u_norm, self.s, self.vt = self.load(svd_path)
        self.tfidf = self.load(tfidf_path)
        self.posts_ids, self.posts_comments = self.load(materials_path)

    def load(self, fpath):
        with open(fpath, "rb") as fin:
            return pickle.load(fin)

    def comments(self, intext):
        try:
            tfidf = self.tfidf
            u_norm, s, vt = self.u_norm, self.s, self.vt        
            sim_vec = self.text_similarity(intext, tfidf, u_norm, s, vt)        
            comments = self.select_comments(sim_vec, 10)
        except Exception as ex:
            print(ex)
            comments = ["..."]
        return comments

    def transform_text(self, intext, tfidf):
        tokens = " ".join(jieba.lcut(intext))
        return tfidf.transform([tokens])

    def text_similarity(self, intext, tfidf, u_norm, s, vt):
        doc_vec = self.transform_text(intext, tfidf).todense()
        doc_proj = np.dot(np.dot(doc_vec, vt.transpose()), np.diag(1/s))
        doc_norm = doc_proj / np.sqrt(np.dot(doc_proj, doc_proj.transpose()))        
        sim = np.dot(doc_norm, u_norm.transpose()).squeeze()
        sim = np.asarray(sim).squeeze()
        return sim

    def select_comments(self, sim_vec, nmax = 10):
        # sample a similar document            
        rs = np.random.RandomState(3234)        
        logits_vec = sim_vec.copy()    
        post_idx = np.argsort(-logits_vec)[:3]    
        # select comments within the selected document    
        sampled = []    
        for post_i in post_idx:        
            sel_text_id = self.posts_ids[post_i]
            sel_comments = self.posts_comments[sel_text_id]
            sampled.extend(sel_comments)    
        rs.shuffle(sampled)
        return sampled[:nmax]