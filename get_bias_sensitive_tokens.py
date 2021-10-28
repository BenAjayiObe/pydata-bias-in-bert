import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

BASE_PATH = "data/embeddings"
EMBEDDINGS_FILE_PATH = f"{BASE_PATH}/BERTLM_ENCODER_LAYER_ONE/bert-base-uncased-embeddings.txt"
TOP_N=100
NUM_PCA_COMPONENTS=10
PAIRS = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"],
         ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["mary", "john"]]


# obtain bert biased tokens
def load_word_vectors(fname):
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.index_to_key)
    return model, vecs, words


def doPCA(pairs, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (a + b) / 2
        norm_a = a - center
        norm_b = b - center
        matrix.append(norm_a)
        matrix.append(norm_b)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components, svd_solver="full")
    pca.fit(matrix)
    return pca


def bias_subspace(embed, num_components=None):
    # define gender direction
    pair = []
    for p in PAIRS:
        pair.append((embed[p[0]], embed[p[1]]))
    pca = doPCA(pair)
    bias_direction = pca.components_[0] # using principle component that has the highest variance in gender bias subspace
    print(f"pca explained variance ratio: {pca.explained_variance_ratio_} \n")
    
    return bias_direction


def data_preprocess():
    embed, vecs, words = load_word_vectors(fname=EMBEDDINGS_FILE_PATH)
    bias_direction = bias_subspace(embed=embed, num_components=NUM_PCA_COMPONENTS)

    bias_sign = bias_direction.dot(embed["woman"] - embed["man"])
    bias_direction = -bias_direction if bias_sign > 0 else bias_direction

    # projection on the gender direction, and got top n biased token
    group1 = embed.similar_by_vector(bias_direction, topn=TOP_N, restrict_vocab=None) # male biased
    group2 = embed.similar_by_vector(-bias_direction, topn=TOP_N, restrict_vocab=None) # female biased
    group3 = embed.similar_by_vector(np.array([0]*embed.vector_size), topn=TOP_N, restrict_vocab=None) # neutral tokens
    male_tokens, male_scores = list(zip(*group1))
    female_tokens, female_scores = list(zip(*group2))
    neutral_tokens, neural_scores = list(zip(*group3))
    
    print(f"TOP 100 MALE SENSITIVE TOKENS \n {male_tokens}\n\n")
    print(f"TOP 100 FEMALE SENSITIVE TOKENS \n {female_tokens}\n\n")
    print(f"TOP 100 NEUTRAL TOKENS \n {neutral_tokens}\n\n")

    np.save(f"{BASE_PATH}/male_bias_sensitive_tokens.npy", np.array([embed[m_token] for m_token in male_tokens]))
    np.save(f"{BASE_PATH}/female_bias_sensitive_tokens.npy", np.array([embed[f_token] for f_token in female_tokens]))
    np.save(f"{BASE_PATH}/neutral_tokens.npy", np.array([embed[n_token] for n_token in neutral_tokens]))


if __name__ == '__main__':
    data_preprocess()
