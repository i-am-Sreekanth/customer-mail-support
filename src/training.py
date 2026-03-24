import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

def train_models(df, save_dir="saved_models"):
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Split features and targets
    X_text = df['text_clean'].tolist()
    
    y_type = df['type']
    y_priority = df['priority']
    y_queue = df['queue']
    
    # Multi-label tags
    mlb = MultiLabelBinarizer()
    y_tags = mlb.fit_transform(df['tags'])
    
    # Sentence Embeddings
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(X_text, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    # Train classifiers
    type_model = LogisticRegression(max_iter=2000)
    type_model.fit(embeddings, y_type)
    
    priority_model = LogisticRegression(max_iter=2000)
    priority_model.fit(embeddings, y_priority)
    
    queue_model = LogisticRegression(max_iter=2000)
    queue_model.fit(embeddings, y_queue)
    
    tag_model = LogisticRegression(max_iter=2000)
    tag_model.fit(embeddings, y_tags)
    
    # Save classifiers
    classifiers = {
        "type_model": type_model,
        "priority_model": priority_model,
        "queue_model": queue_model,
        "tag_model": tag_model
    }
    joblib.dump(classifiers, f"{save_dir}/classifiers.pkl")
    
    # Save mlb separately
    joblib.dump(mlb, f"{save_dir}/mlb.pkl")
    
    # Save embeddings + model
    np.save(f"{save_dir}/train_embeddings.npy", embeddings)
    embed_model.save(f"{save_dir}/embedding_model")
    
    # Save training df
    df.to_csv(f"{save_dir}/train_df.csv", index=False)
    
    print("Training complete. All models saved to:", save_dir)
    
    return classifiers, mlb, embed_model, embeddings