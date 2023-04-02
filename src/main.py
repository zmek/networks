import pandas as pd
import csv
import numpy as np

from functions.embeddings_using_pd import get_word_embeddings
from functions.embeddings_viz_images import reduce_and_draw_network_map_with_images


# Specify input data
filename = "data-raw/Tom LD _ Life Story Presentation.txt"

# Specify model path
model_path = "GloVE/glove.6B/glove.6B.300d.txt"


def main(filename, model_path):
    words = pd.read_table(
        model_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
    )

    model_df = pd.read_table(model_path, sep=" ", index_col=0, header=None, quoting=3)
    model_dict = {
        word: embeddings for word, embeddings in zip(model_df.index, model_df.values)
    }

    em = get_word_embeddings(filename, model_path)

    reduce_and_draw_network_map_with_images(
        np.array(em[1:]), "img", "Happy Birthday Tom", handle_isolates=False
    )


main(filename, model_path)
