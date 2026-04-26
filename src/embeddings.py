# embeddings.py

import numpy as np

def load_glove_embeddings(glove_path, word2idx, dim=300):
    embedding_matrix = np.zeros((len(word2idx), dim))

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word in word2idx:
                embedding_matrix[word2idx[word]] = \
                    np.array(values[1:], dtype='float32')

    return embedding_matrix
