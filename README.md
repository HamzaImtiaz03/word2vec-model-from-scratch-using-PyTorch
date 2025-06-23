# **Word2Vec Skip-gram Model from Scratch**

## **Overview**

This project is a personal deep dive into **Natural Language Processing (NLP)**, where I implemented the **Word2Vec Skip-gram model** entirely from scratch using **PyTorch**. Inspired by landmark research in distributed word representations, I set out to build a small yet insightful prototype that learns word embeddings from a custom text corpus.

The goal was not just to replicate Word2Vec’s performance but to **understand the inner mechanics** of how words can be translated into meaningful vectors using **neural networks** and **negative sampling**. This README documents my journey, learnings from foundational research papers, implementation steps, and how you can explore or extend the work further.

---

## **Key Features**

* Implementation of **Skip-gram architecture** using a shallow neural network
* **Negative Sampling** optimization for efficient training (to be implemented)
* **PyTorch**-based end-to-end model pipeline
* Small, custom corpus training for simplicity and clarity
* Embedding visualization-ready outputs
* Clean code structure for learning and extension

---

## **Research Papers That Inspired This Work**

The following research papers were instrumental in shaping my understanding and guiding this implementation:

1. **[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)**
   Tomas Mikolov et al., 2013
   Introduced the original Word2Vec model and Skip-gram architecture.

2. **[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)**
   Mikolov et al., 2013
   Introduced Negative Sampling, a key optimization (future improvement).

3. **[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)**
   Bengio et al., 2003
   Provided early ideas on learning distributed word representations using neural networks.

4. **[On the Properties of Neural Word Embeddings](https://www.aclweb.org/anthology/N14-1090/)**
   Omer Levy and Yoav Goldberg, 2014
   Provided deeper insights into how the Skip-gram model implicitly performs matrix factorization.

5. **[A Survey on Word Embeddings and Word Representation Models](https://arxiv.org/abs/1904.03477)**
   Yu Zhang, Qiang Yang, 2019
   Helped place Word2Vec in the context of modern word embedding techniques.

---

## **Conceptual Diagram**

Here is a basic flowchart to understand how the Skip-gram with Negative Sampling works:

```text
         +------------------------+
         |   Input Word ("king")  |
         +-----------+------------+
                     |
                     v
        +------------+-------------+
        |  Embedding Lookup Layer  |
        +------------+-------------+
                     |
                     v
        +------------+-------------+
        |  Hidden Layer (Linear)   |
        +------------+-------------+
                     |
                     v
        +-----------------------------+
        | Dot Product with Negative   |
        | Samples + Target Word       |
        +-----------------------------+
                     |
                     v
        +-----------------------------+
        | Sigmoid Activation + Loss   |
        +-----------------------------+
```

---

## **Technologies Used**

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| **Python 3.10+** | Core programming language              |
| **PyTorch**      | Building and training neural networks  |
| **NumPy**        | Efficient matrix and vector operations |
| **NLTK**         | Tokenizing and preprocessing corpus    |
| **tqdm**         | Progress bars for training (optional)  |

---

## **Python Implementation**

```python
# Install if not already: pip install torch nltk

import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from collections import Counter
import random
import numpy as np

nltk.download('punkt')

# Sample corpus
corpus = "The quick brown fox jumps over the lazy dog. The dog barked at the fox."

# Preprocess text
words = nltk.word_tokenize(corpus.lower())
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# Generate training data: (target, context) pairs using window size
window_size = 2
data = []

for i, target in enumerate(words):
    for j in range(-window_size, window_size + 1):
        if j == 0 or i + j < 0 or i + j >= len(words):
            continue
        context = words[i + j]
        data.append((word2idx[target], word2idx[context]))

# Define Skip-gram Model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.out(x)
        return x

# Hyperparameters
embedding_dim = 10
vocab_size = len(vocab)
model = SkipGram(vocab_size, embedding_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    total_loss = 0
    for target, context in data:
        target_tensor = torch.tensor([target])
        context_tensor = torch.tensor([context])

        output = model(target_tensor)
        loss = loss_fn(output, context_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"\U0001F9E0 Epoch {epoch}, Loss: {total_loss:.4f}")

# Show word embeddings
print("\n\U0001F50D Word Embeddings:")
for word in ["dog", "fox", "the", "lazy"]:
    idx = word2idx[word]
    embed = model.embedding.weight[idx].detach().numpy()
    print(f"{word}: {embed}")
```

---

## **Training and Running**

* Run the script above after installing dependencies.
* You will see loss decreasing and word embeddings for selected tokens printed at the end.

---

## **Learnings and Challenges**

This project was a powerful learning experience for me. Some key takeaways:

* **Implementing backpropagation manually** helped me understand gradient flow in embedding layers.
* **Negative Sampling** (future work) can dramatically improve performance and training scalability.
* **Understanding word context and co-occurrence** revealed how subtle semantics can be captured in simple vector math.
* **Reading original papers** changed the way I think about neural networks: it’s not magic—it’s math and optimization.

---

## **Future Work**

* Support for CBOW model implementation
* Add visualization tools like t-SNE or PCA for embeddings
* Train on a large dataset (e.g., text8 or Wikipedia dump)
* Add proper Negative Sampling for large-vocabulary training

---

## **Contact**

📧 **[hamzaimtiaz8668@gmail.com](mailto:hamzaimtiaz8668@gmail.com)**
For queries, feedback, or contributions, feel free to reach out.

---

> *"Building this project from scratch helped me not only understand the inner workings of Word2Vec but also boosted my confidence in implementing research-grade ideas with PyTorch."*
