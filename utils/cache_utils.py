import os
import logging
import pandas as pd

def ensure_cache_directory(cache_dir):
    """Ensure cache directory exists and is writable."""
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return os.access(cache_dir, os.W_OK)
    except Exception as e:
        logging.error(f"Erreur lors de la création du répertoire de cache: {e}")
        return False

def save_to_cache(df, cache_file, meta_file=None, meta_data=None):
    """Save DataFrame and optional metadata to cache."""
    try:
        df.to_csv(cache_file, index=False)
        if meta_file and meta_data:
            with open(meta_file, 'w') as f:
                f.write(str(meta_data))
        return True
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde dans le cache: {e}")
        return False

def load_from_cache(cache_file, meta_file=None):
    """Load DataFrame and optional metadata from cache."""
    try:
        df = pd.read_csv(cache_file)
        meta_data = None
        if meta_file and os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta_data = f.read().strip()
        return df, meta_data
    except Exception as e:
        logging.error(f"Erreur lors du chargement depuis le cache: {e}")
        return None, None