from sentence_transformers import SentenceTransformer

class BertEmbedder:

    def __init__(self):
        print("[INFO] Loading BERT model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode_documents(self, documents):
        """
        Generate embeddings for a list of documents
        """
        embeddings = self.model.encode(documents)
        return embeddings

    def encode_single(self, document):
        """
        Generate embedding for single document
        """
        embedding = self.model.encode([document])
        return embedding