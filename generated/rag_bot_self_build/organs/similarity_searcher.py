from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def similarity_searcher(chunks, query_vector):
    """
    :param chunks: list of dictionaries containing 'vector' key for each chunk
    :param query_vector: vector representation of the query
    :return: dictionary with keys 'top_chunks' containing the top 3 most similar chunks
    """
    similarities = []
    for chunk in chunks:
        sim = cosine_similarity([query_vector], [chunk['vector']])
        similarities.append(sim[0][0])

    sorted_indices = np.argsort(similarities)[::-1]
    top_chunks = [chunks[i] for i in sorted_indices[:3]]

    return {"top_chunks": top_chunks}
