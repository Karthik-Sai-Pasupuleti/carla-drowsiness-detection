# vector_db.py

import faiss
import numpy as np
import pickle
import os
import time
from collections import Counter

class PrefixVectorDB:
    """
    A lightweight FAISS-based vector database for storing, searching, and managing
    low-dimensional prefix vectors along with associated metadata such as interventions
    and sources.

    This class supports:
    - Normalization of input vectors
    - Cosine distance checks for deduplication
    - Automatic removal of redundant entries to cap DB size
    - Search and retrieval of similar vectors using FAISS

    Attributes:
        dim (int): Dimensionality of the feature vectors (default is 9).
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the pickle file containing metadata.
        index (faiss.IndexIDMap): FAISS index for vector storage and search.
        metadata (dict): Dictionary storing metadata for each vector ID.
        current_id (int): Next available ID for a new vector.
    """

    def __init__(self, dim=9, index_path='index.faiss', metadata_path='metadata.pkl'):
        """
        Initializes the PrefixVectorDB. Loads existing FAISS index and metadata if available,
        or creates new ones if not found.

        Args:
            dim (int): Dimension of the feature vectors (default: 9).
            index_path (str): File path to save/load FAISS index.
            metadata_path (str): File path to save/load metadata.
        """
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

        self.current_id = max(self.metadata.keys(), default=0) + 1

    def normalize_vector(self, vector):
        """
        Normalizes a vector using L2 norm.

        Args:
            vector (np.ndarray): Input vector.

        Returns:
            np.ndarray: Normalized vector. Returns original vector if norm is zero.
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def cosine_distance(self, v1, v2):
        """
        Calculates cosine distance between two vectors.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            float: Cosine distance (1 - cosine similarity).
        """
        v1 = self.normalize_vector(v1)
        v2 = self.normalize_vector(v2)
        return 1 - np.dot(v1, v2)

    def add_vector(self, feature_vector: np.ndarray, intervention: str, source: str):
        """
        Adds a feature vector with associated intervention and source to the database.
        Performs duplicate checks and enforces a maximum size of 2000 entries.

        Args:
            feature_vector (np.ndarray): Vector of shape (9,).
            intervention (str): Label/action to store with the vector.
            source (str): Source of the vector (e.g., 'inference').

        Raises:
            AssertionError: If the feature vector shape is not (9,).
        """
        assert feature_vector.shape == (9,), f"Expected shape (9,), got {feature_vector.shape}"
        if intervention.lower() in ["none", "", "no action", "driver alert"]:
            return  # Skip irrelevant samples

        vector = self.normalize_vector(feature_vector)

        if len(self.metadata) > 0:
            D, I = self.index.search(np.expand_dims(vector, axis=0), k=min(5, len(self.metadata)))
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                neighbor = self.metadata.get(int(idx), {})
                try:
                    reconstructed = self.index.reconstruct(int(idx))
                    cosine_dist = self.cosine_distance(vector, reconstructed)
                except Exception:
                    continue
                if neighbor.get("intervention") == intervention and cosine_dist < 0.05:
                    return  # Skip duplicate

        if len(self.metadata) >= 2000:
            D, I = self.index.search(np.expand_dims(vector, axis=0), k=10)
            close_ids = [i for i, d in zip(I[0], D[0]) if i != -1 and d < 0.1]
            if close_ids:
                remove_id = close_ids[0]
            else:
                counter = Counter([v['intervention'] for v in self.metadata.values()])
                most_common = counter.most_common(1)[0][0]
                remove_id = next(k for k, v in self.metadata.items() if v['intervention'] == most_common)

            self.index.remove_ids(np.array([remove_id]))
            del self.metadata[remove_id]

        self.index.add_with_ids(np.expand_dims(vector, axis=0), np.array([self.current_id]))
        self.metadata[self.current_id] = {
            'intervention': intervention,
            'source': source,
            'timestamp': time.time(),
            'last_accessed': time.time()
        }
        self.current_id += 1

    def save(self):
        """
        Saves the current FAISS index and metadata dictionary to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search(self, query_vector: np.ndarray, k=5):
        """
        Searches for the top-k most similar vectors to the query vector.

        Args:
            query_vector (np.ndarray): Input vector of shape (9,).
            k (int): Number of nearest neighbors to return (default: 5).

        Returns:
            List[Tuple[int, dict, float]]: List of tuples containing (vector_id, metadata, distance).
        """
        assert query_vector.shape == (9,), f"Expected shape (9,), got {query_vector.shape}"
        query_vector = self.normalize_vector(query_vector)
        D, I = self.index.search(np.expand_dims(query_vector, axis=0), k)
        results = []
        for i, d in zip(I[0], D[0]):
            if i == -1:
                continue
            self.metadata[i]['last_accessed'] = time.time()
            results.append((int(i), self.metadata.get(int(i), {}), float(d)))
        return results


# === Inference-Time Helper Functions ===

def runtime_add(feature_vector: np.ndarray, intervention: str):
    """
    Adds a vector with its intervention label to the database at inference time.

    Args:
        feature_vector (np.ndarray): Feature vector to add.
        intervention (str): Intervention label to associate.
    """
    db = PrefixVectorDB()
    db.add_vector(feature_vector, intervention, source='inference')
    db.save()

def retrieve_similar_vectors(feature_vector: np.ndarray, k=5):
    """
    Retrieves k most similar vectors from the database for the given feature vector.

    Args:
        feature_vector (np.ndarray): Query vector.
        k (int): Number of similar vectors to retrieve (default: 5).

    Returns:
        List[Tuple[int, dict, float]]: List of (vector_id, metadata, distance).
    """
    db = PrefixVectorDB()
    return db.search(feature_vector, k)
