import numpy as np
import math
from collections import Counter
import json


documents = [
    "YOLOv10 là mô hình phát hiện vật thể",
    "RAG là kỹ thuật tăng cường sinh văn bản",
    "YOLO là real-time object detection",
    "RAG kết hợp retrieval và generation",
    "Chunking chia nhỏ văn bản thành các đoạn",
    "Embedding chuyển text thành vector số"
]
print(f"Tổng số documents: {len(documents)}")

def tokenizer(text):
    text = text.lower()

    for punct in [',', '.', '!', '?', ';', ':', '"', "'"]:
        text.replace(punct,'')
    return text.split()

print(tokenizer(documents[0]))

class BM25Scratch:
    def __init__(self, documents, alpha = 0.5, k = 4):
        self.documents = documents
        self.alpha = alpha
        self.k = k

        self.tokenized_docs = [tokenizer(doc) for doc in self.documents]
        self.doc_length = [len(doc) for doc in self.documents]
        self.avg_length = sum(self.doc_length) / len(self.doc_length)
    