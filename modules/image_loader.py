
import sqlite3
from typing import List
from . import datasetHelper
from sentence_transformers import SentenceTransformer, util 
import torch
import numpy as np


class ImageLoader: 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    con = sqlite3.connect("fashion_ai.db", check_same_thread=False)
    
    def __init__(self, data_file: str): 
        self.cur = self.con.cursor()
        self.df, self.vocabulary = datasetHelper.createVocabulary(datasetHelper.createDataset(data_file))
        self.corpus_embeddings = [np.frombuffer(i[0], dtype=np.float32) for i in self.cur.execute("SELECT embedding FROM images")]
        
    # Returns image blob 
    
    def similarity_search(self, item_output: str) -> str:
        query_embedding = self.model.encode(item_output, normalize_embeddings=True)
        dot_scores = util.dot_score(query_embedding, self.corpus_embeddings)[0]
        top_result = torch.topk(dot_scores, k=5)
        best_result = int(top_result[1][0])
        print(best_result)
        best_row = self.df[self.df["item_name"] == self.vocabulary[best_result]]
        db_key = str(best_row["item_imageid"].iloc[0])
        print(db_key)
        return self.image_query(db_key)
        
    def image_query(self, set_id: str) -> str: 
        query = "SELECT * FROM images WHERE set_id = ?"
        self.cur.execute(query, (set_id,))
        rows = self.cur.fetchall() 
        return rows[0]
            
    def get_all_images(self, limit: int, offset: int) -> List[object]: 
        self.cur.execute("SELECT * FROM images LIMIT ? OFFSET ?", (limit, offset))
        rows = self.cur.fetchall()
        return rows
    