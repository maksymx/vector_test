import os

import faiss
import wikipediaapi
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-MiniLM-L6-v2"

app = Flask(__name__)


def retrieve_articles(article_titles, cache_dir):
    wiki_wiki = wikipediaapi.Wikipedia(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    articles = {}
    for title in article_titles:
        page = wiki_wiki.page(title)
        text = page.text
        articles[title] = text
        # Write text content to a text file
        file_path = os.path.join(cache_dir, f"{title}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    return articles


def chunk_text(text, chunk_size=500):
    # Split text into chunks of chunk_size words
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def vectorize_text(texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts)
    return embeddings

def store_embeddings_in_database(embeddings):
    # Implement storing embeddings in a graph database (e.g., Neo4j, ArangoDB, etc.)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Create an index
    index.add(embeddings)  # Add vectors to the index
    return index


def search(query, index, k=5):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return distances, indices


@app.route("/search", methods=["GET"])
def search_handler():
    query = request.args.get("query")
    if query:
        results = []
        for title, idx in index.items():
            distances, indices = search(query, idx)
            for dist, _idx in zip(distances[0], indices[0]):
                results.append(
                    {
                        "distance": float(str(dist)),
                        "title": title,
                        "text_chunk": str(chunks[title][_idx]),
                    }
                )
        return jsonify({"results": results})
    else:
        return jsonify({"error": 'Query parameter "query" is required.'}), 400


if __name__ == "__main__":
    article_titles = ['Eastern_chipmunk', 'Owl', 'Inverness', 'Londinium', 'Kyiv', 'Dog']
    cache_dir = "article_cache"  # Directory to store cached articles
    os.makedirs(cache_dir, exist_ok=True)

    articles = retrieve_articles(article_titles, cache_dir)

    chunks = {}
    for title, text in articles.items():
        chunks[title] = chunk_text(text)

    embeddings = {}
    for title, chunk_list in chunks.items():
        embeddings[title] = vectorize_text(chunk_list)

    index = {}
    for title, embedding in embeddings.items():
        index[title] = store_embeddings_in_database(embedding)

    titles = list(articles.keys())

    app.run(host="0.0.0.0", port=5000)
