import numpy as np

def preprocess(text):

    """Preprocess the input text to create a corpus and mappings from words to IDs and vice versa."""

    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word2id, id2word = {}, {}
    for word in words:
        if word not in word2id:
            new_id = len(word2id)
            word2id[word] = new_id
            id2word[new_id] = word
    
    corpus = np.array([word2id[word] for word in words])

    return corpus, word2id, id2word


# # preprocess function Example usage
# if __name__ == "__main__":
#     text = "You say goodbye and I say hello."
#     corpus, word2id, id2word = preprocess(text)
    
#     print("Corpus:", corpus)
#     print("Word to ID mapping:", word2id)
#     print("ID to Word mapping:", id2word)


def create_co_matrix(corpus, vocab_size, window_size = 1):

    """Create a co-occurrence matrix from the corpus."""

    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix


def cos_similarity(x, y, eps = 1e-8):

    """Calculate the cosine similarity between two vectors."""

    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)  
    return np.dot(nx, ny)


def most_similar(query, word2id, id2word, co_matrix, top=5):
    """Find the most similar words to the query word based on the co-occurrence matrix."""

    if query not in word2id:
        print(f"{query} is not found in the vocabulary.")
        return None
    
    query_id = word2id[query]
    query_vec = co_matrix[query_id]

    similarity = np.zeros(len(id2word))
    for i in range(len(id2word)):
        similarity[i] = cos_similarity(query_vec, co_matrix[i])
    
    count = 0
    for i in (-1 * similarity).argsort():
        if id2word[i] == query:
            continue
        print(f"{count + 1}: {id2word[i]}: {similarity[i]:.4f}")
        count += 1
        if count >= top:
            return


def ppmi(C, verbose = False, eps = 1e-8):
    """
    Calculate the Positive Pointwise Mutual Information (PPMI) matrix from the co-occurrence matrix C."""

    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2((C[i, j] * N) / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print(f"{cnt / total * 100:.1f}% done")
    return M


def create_contexts_target(corpus, window_size=1):
    """
    Create contexts and target words from the corpus for training a CBOW model.
    Each context is a list of words surrounding the target word within the specified window size.
    """

    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    """Convert the corpus of word IDs into one-hot encoded vectors."""
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate