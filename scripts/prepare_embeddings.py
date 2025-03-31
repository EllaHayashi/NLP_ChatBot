# scripts/prepare_embeddings.py

#this script helps build a semantic search index using sentence embeddings and FAISS
#that means we take the 'context' or bible chunks and convert them into embeddings for fast retrieval
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# load sentence embedding model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# open cleaned squad file and extract all the contexts
with open("data/squad/clean_train.json", "r") as f:
    data = json.load(f)

contexts = []
for item in data["data"]:
    for para in item["paragraphs"]:
        context = para["context"]
        contexts.append(context)

# encode the contexts into embeddings
embeddings = encoder.encode(contexts, convert_to_numpy=True, show_progress_bar=True)

#build a faiss index for lookup
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# save the index + context
faiss.write_index(index, "retrieval/context.index")
with open("retrieval/contexts.json", "w") as f:
    json.dump(contexts, f)

print(f"共保存 {len(contexts)} 段 context 到索引中")
