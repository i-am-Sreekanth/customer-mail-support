import pandas as pd

def load_dataset(path):
    """Load the dataset CSV/TSV"""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    if 'language' in df.columns:
        df = df[df['language'] == 'en']
    if 'version' in df.columns:
        df = df.drop(columns=['version'])
    
    # Create a combined text column
    df['text_clean'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df['text_clean'] = df['text_clean'].str.lower().str.strip()
    
    # Ensure tags are lists
    tag_cols = [col for col in df.columns if col.startswith('tag_')]
    def combine_tags(row):
        return [row[col] for col in tag_cols if pd.notna(row[col])]
    
    df['tags'] = df.apply(combine_tags, axis=1)
    
    return df

def save_clean_data(df, path):
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to {path}")