# scripts/qa_retrieval.py
# this is the RAG system
# it searches for the most relevant paragraph based on the embeded context
# then uses fine tuned AQ model to extract the answer from that context
#e.g. given a question, use FAISS to get most relevant context and then extract the answere

import torch
import faiss
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# fine tuned model path
model_path = "models/distilbert-qa"
qa_tokenizer = AutoTokenizer.from_pretrained(model_path)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# load encoder and faiss encoded context
encoder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("retrieval/context.index")
with open("retrieval/contexts.json", "r") as f:
    all_contexts = json.load(f)

# main question function where you encode the question with faiss and then search for most similar faiss embedding we have
def qa_with_auto_context(question, top_k=1):
    # encode question
    q_vec = encoder.encode([question], convert_to_numpy=True)

    # search for most relevant context
    D, I = index.search(q_vec, top_k)
    context = all_contexts[I[0][0]]

    # run QA model
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    # extract best answere span
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    #convert tokensinto human readbale because words might be split up
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

    #returns everything
    return {
        "question": question,
        "context": context,
        "answer": answer.strip()
    }

# main question
if __name__ == "__main__":
    q = "What did God create?"
    result = qa_with_auto_context(q)
    print(f"{result['question']}")
    print(f"Context: {result['context'][:100]}...")
    print(f"Answer: {result['answer']}")
