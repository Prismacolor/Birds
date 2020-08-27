import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize

# pull in the data and get the comments
in_birds = pd.read_csv('datasets\\US-IN_data_subset.txt',  encoding='latin-1', sep='\t')
ky_birds = pd.read_csv('datasets\\US-KY_data_subset.txt',  encoding='latin-1', sep='\t')
oh_birds = pd.read_csv('datasets\\US-OH_data_subset.txt',  encoding='latin-1', sep='\t')
tx_birds = pd.read_csv('datasets\\US-TX_data_subset.txt',  encoding='latin-1', sep='\t', nrows=50)

master_birds_df = pd.concat([in_birds, ky_birds, oh_birds])
# print(master_birds_df.columns)
test_birds_df = pd.DataFrame(tx_birds)

comments_df = master_birds_df[['TRIP COMMENTS']]
comments_df = comments_df.dropna()

test_comments = test_birds_df[['TRIP COMMENTS']]
test_comments = test_comments.dropna()


# tokenize and remove stopwords from text
def tokenize_words(comment_df):
    stop = stopwords.words('english')
    clean_text = []

    for row in comment_df['TRIP COMMENTS']:
        cleaned_row = [word for word in word_tokenize(row.lower()) if word not in stop]
        clean_text.append(cleaned_row)

    comment_df['Tokenized Comments'] = clean_text
    # print(comments_df.head())

    return comment_df


# try vectorization using TF IDF
def fit_vectorizer(cleaned_df):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=250)
    X = vectorizer.fit_transform(cleaned_df['TRIP COMMENTS'])

    return vectorizer, X


# find the ideal K
def create_clusters(X):
    distortions = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.figure(0)
    plt.plot(range(1, 11), distortions, marker=0)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('plots\\ideal_k.png')

    # the result of this plot shows that eight or nine might be a solid number of clusters for this set TR
    final_K = 9
    # final_K = 5

    return final_K


# create the final model and plot it
def final_k_model(vectorizer, X, finalk):
    count = 0
    final_k_mod = KMeans(n_clusters=finalk, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    final_k_mod.fit(X)

    # plot the results:
    centroids = final_k_mod.cluster_centers_

    # give some idea what words are in what clusters
    for centroid in centroids:
        count += 1
        score_this_centroid = {}
        for word in vectorizer.vocabulary_.keys():
            score_this_centroid[word] = centroid[vectorizer.vocabulary_[word]]
        print(count, score_this_centroid)
        # print(score_this_centroid)

    tsne_init = 'pca'
    tsne_perplexity = 20.0
    tsne_early_exaggeration = 4.0
    tsne_learning_rate = 1000
    random_state = 1
    tsnemodel = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
                 early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

    transformed_centroids = tsnemodel.fit_transform(centroids)

    plt.figure(1)
    plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
    plt.savefig('plots\\cluster.png')
    plt.show()

    print(final_k_mod)
    return final_k_mod


def test_final_k(model, test_set):
    model.predict(test_set)

    print(model.labels_)

    final_score = silhouette_score(test_set, labels=model.predict(test_set))
    print(final_score)


cleaned_comments_df = tokenize_words(comments_df)
vect, X_train = fit_vectorizer(cleaned_comments_df)
final_k = create_clusters(X_train)
final_model = final_k_model(vect, X_train, final_k)

cleaned_test = tokenize_words(test_comments)
vect, test_train = fit_vectorizer(cleaned_test)
test_final_k(final_model, test_train)

'''The final silhouette score was 0.05 which indicates that there are overlapping clusters. A score 
near 0 indicates overlap. A score near -1 indicates items being placed in the wrong clusters. A score
near 1 means the clusters are clearly defined. A reduction of k did not yield better results'''