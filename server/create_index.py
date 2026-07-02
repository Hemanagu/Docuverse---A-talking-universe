from rag.embedder import create_vector_store

chunks = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Supervised learning uses labeled data.",
    "Unsupervised learning finds hidden patterns in data.",
    "Reinforcement learning learns by rewards."
]

create_vector_store(chunks, "test")

print("Vector index created successfully!")