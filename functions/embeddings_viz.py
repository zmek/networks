import numpy as np
from sklearn.manifold import TSNE
import graphviz as gv


def reduce_and_draw_network_map(embeddings, file_path):
    # Perform t-SNE to reduce the embeddings to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a Graphviz graph object
    graph = gv.Graph(engine="dot")

    # Add nodes to the graph
    for i in range(len(embeddings_2d)):
        graph.node(str(i), pos=f"{embeddings_2d[i,0]},{embeddings_2d[i,1]}")

    # Add edges to the graph
    for i in range(len(embeddings_2d)):
        for j in range(i + 1, len(embeddings_2d)):
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if similarity > 0.2:
                graph.edge(str(i), str(j), weight=str(similarity))

    # Draw the network map
    graph.format = "png"
    graph.render(filename=file_path)

    return graph
