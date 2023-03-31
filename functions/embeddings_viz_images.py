import numpy as np
from sklearn.manifold import TSNE
import graphviz as gv
from PIL import Image


def reduce_and_draw_network_map_with_images(
    embeddings, image_folder="img", _similarity=0.5
):
    # Perform t-SNE to reduce the embeddings to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a Graphviz graph object
    graph = gv.Graph(engine="neato")

    # Add nodes to the graph
    for i in range(len(embeddings_2d)):
        # Load image
        img_path = f"{image_folder}/Slide{i+1}.jpeg"
        # img = Image.open(img_path)  # .resize((50, 50))

        # Add node with image
        with graph.subgraph(name=f"cluster_{i}") as c:
            c.attr(label=f"{i}")
            c.attr(fontsize="10")
            c.attr(style="filled")
            c.attr(color="white")
            c.node(f"node{i}", image=img_path, shape="none", label="")

            # Set position of cluster
            c.attr(pos=f"{embeddings_2d[i,0]},{embeddings_2d[i,1]}!")

    # Add edges to the graph
    for i in range(len(embeddings_2d)):
        for j in range(i + 1, len(embeddings_2d)):
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if similarity > 0.8:
                graph.edge(f"node{i}", f"node{j}", weight=str(similarity))

    # Draw the network map
    graph.format = "png"
    graph.render(filename="network_map_with_images 3")
