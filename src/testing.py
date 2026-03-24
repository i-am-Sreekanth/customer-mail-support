import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TicketPredictor:
    def __init__(self, model_dir="saved_models"):
        # Load classifiers
        classifiers = joblib.load(f"{model_dir}/classifiers.pkl")
        self.type_model = classifiers['type_model']
        self.priority_model = classifiers['priority_model']
        self.queue_model = classifiers['queue_model']
        self.tag_model = classifiers['tag_model']
        
        # Load MultiLabelBinarizer
        self.mlb = joblib.load(f"{model_dir}/mlb.pkl")
        
        # Load embedding model + precomputed embeddings
        self.embed_model = SentenceTransformer(f"{model_dir}/embedding_model")
        self.train_embeddings = np.load(f"{model_dir}/train_embeddings.npy")
        
        # Load training dataframe for retrieval
        self.train_df = pd.read_csv(f"{model_dir}/train_df.csv")
    
    def retrieve_best_answer(self, query):
        query_emb = self.embed_model.encode([query])
        sims = cosine_similarity(query_emb, self.train_embeddings)[0]
        top_idx = sims.argmax()
        best_match = self.train_df.iloc[top_idx]
        return {
            "similarity_score": float(sims[top_idx]),
            "matched_text": best_match['text_clean'],
            "answer": best_match['answer']
        }
    
    def predict_ticket(self, text):
        text_clean = text.lower()
        vec = self.embed_model.encode([text_clean])
        
        best_match = self.retrieve_best_answer(text)
        
        result = {
            "type": self.type_model.predict(vec)[0],
            "priority": self.priority_model.predict(vec)[0],
            "queue": self.queue_model.predict(vec)[0],
            "tags": self.mlb.inverse_transform(self.tag_model.predict(vec))[0],
            "best_match": best_match
        }
        return result
    
    def pretty_print(self, result, query):
        print("\n===== YOUR QUERY =====\n")
        print(query)
        print("\n===== PREDICTION =====\n")
        print(f"Type      : {result['type']}")
        print(f"Priority  : {result['priority']}")
        print(f"Queue     : {result['queue']}")
        print(f"Tags      : {', '.join(result['tags'])}")
        print("\n===== MATCHED PAST CASE =====\n")
        print(result['best_match']['matched_text'])
        print("\n===== SUGGESTED RESPONSE =====\n")
        print(result['best_match']['answer'])
        print("\n===== CONFIDENCE =====")
        print(f"Similarity Score: {result['best_match']['similarity_score']:.3f}")