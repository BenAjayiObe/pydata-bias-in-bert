# Some codes are from https://github.com/shauli-ravfogel/nullspace_projection

import sys
sys.path.append("nullspace_projection/src")
sys.path.append("nullspace_projection/")
import debias
import random
import sklearn
import numpy as np
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

GENDER_CLF = LinearSVC
GENDER_CLF_PARAMS = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}
NUM_CLASSIFIERS = 25
EMBEDDINGS_SIZE = 768
IS_AUTOREGRESSIVE = True
MIN_ACCURACY = 0
DROPOUT_RATE = 0
TEST_SET_RATIO = 0.3
RANDOM_STATE = 42


def split_dataset(male_feat, female_feat, neut_feat):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    X = np.concatenate((male_feat, female_feat, neut_feat), axis=0)
    y_masc = np.ones(male_feat.shape[0], dtype=int)
    y_fem = np.zeros(female_feat.shape[0], dtype=int)
    y_neut = -np.ones(neut_feat.shape[0], dtype=int)
    y = np.concatenate((y_masc, y_fem, y_neut))
    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=TEST_SET_RATIO, random_state=RANDOM_STATE
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=TEST_SET_RATIO, random_state=RANDOM_STATE
    )
    print("Train size: {}; Dev size: {}; Test size: {}".format(X_train.shape[0], X_dev.shape[0], X_test.shape[0]))
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def debias_effect_analysis(P, X_test, Y_test):
    def tsne(vecs, labels, title="", ind2label=None):
        tsne = TSNE(n_components=2, perplexity=50, metric="cosine", n_iter=3000, square_distances=True)
        vecs_2d = tsne.fit_transform(vecs)
        label_names = sorted(list(set(labels.tolist())))
        names = sorted(set(labels.tolist()))

        plt.figure(figsize=(6, 5))
        colors = "red", "blue"
        for i, c, label in zip(sorted(set(labels.tolist())), colors, names):
            plt.scatter(vecs_2d[labels == i, 0], vecs_2d[labels == i, 1], c=c,
                        label=label if ind2label is None else ind2label[label], alpha=0.3,
                        marker="s" if i == 0 else "o")
            plt.legend(loc="upper right")

        plt.title(title)
        plt.savefig("images/embeddings-{}.png".format(title), dpi=600)
        plt.show()
        return vecs_2d

    all_significantly_biased_vecs = np.concatenate((male_feat, female_feat))
    all_significantly_biased_labels = np.concatenate(
        (np.ones(male_feat.shape[0], dtype=int), np.zeros(female_feat.shape[0], dtype=int)))
    ind2label = {1: "Male-biased", 0: "Female-biased"}
    tsne_before = tsne(all_significantly_biased_vecs, all_significantly_biased_labels,
                       ind2label=ind2label, title="before-null-projection")

    all_significantly_biased_cleaned = P.dot(all_significantly_biased_vecs.T).T
    tsne_after = tsne(all_significantly_biased_cleaned, all_significantly_biased_labels,
                      ind2label=ind2label, title="after-null-projection")

    def compute_v_measure(vecs, labels_true, k=2):
        np.random.seed(0)
        clustering = sklearn.cluster.KMeans(n_clusters=k)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        return sklearn.metrics.v_measure_score(labels_true, labels_pred)

    # remove neutral class, keep only male and female biased
    X_test = X_test[Y_test != -1]
    Y_test = Y_test[Y_test != -1]
    X_test_cleaned = (P.dot(X_test.T)).T

    print("V-measure-before (TSNE space): {}".format(compute_v_measure(tsne_before, all_significantly_biased_labels)))
    print("V-measure-after (TSNE space): {}".format(compute_v_measure(tsne_after, all_significantly_biased_labels)))

    print("V-measure-before (original space): {}".format(
        compute_v_measure(all_significantly_biased_vecs, all_significantly_biased_labels), k=2))
    print("V-measure-after (original space): {}".format(compute_v_measure(X_test_cleaned, Y_test), k=2))


if __name__ == '__main__':
    # we load the saved context embedding
    male_feat, female_feat, neut_feat = np.load("data/embeddings/male_bias_sensitive_tokens.npy"), \
                                        np.load("data/embeddings/female_bias_sensitive_tokens.npy"), \
                                        np.load("data/embeddings/neutral_tokens.npy")

    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = split_dataset(male_feat, female_feat, neut_feat)
    P, rowspace_projs, Ws = debias.get_debiasing_projection(classifier_class=GENDER_CLF,
                                                            cls_params=GENDER_CLF_PARAMS,
                                                            num_classifiers=NUM_CLASSIFIERS, 
                                                            input_dim=EMBEDDINGS_SIZE,
                                                            is_autoregressive=IS_AUTOREGRESSIVE,
                                                            min_accuracy=MIN_ACCURACY,
                                                            X_train=X_train,
                                                            Y_train=Y_train,
                                                            X_dev=X_dev,
                                                            Y_dev=Y_dev,
                                                            Y_train_main=None,
                                                            Y_dev_main=None,
                                                            by_class=False,
                                                            dropout_rate=DROPOUT_RATE)
    debias_effect_analysis(P, X_test, Y_test)
    np.save("data/nullspace_vector.npy", P)
