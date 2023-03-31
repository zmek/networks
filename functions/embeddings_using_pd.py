# This file was generated by ChatGPT

import numpy as np
import pandas as pd


def get_word_embeddings(filename, model_path):
    # Load the pre-trained GloVe model from the .txt file using pandas
    model_df = pd.read_table(model_path, sep=" ", index_col=0, header=None, quoting=3)
    model_dict = {
        word: embeddings for word, embeddings in zip(model_df.index, model_df.values)
    }

    # Open the input file and read its contents
    with open(filename, "r") as f:
        contents = f.read()

    # Split the contents into blocks based on integer line followed by a blank line
    blocks = contents.split("\n\n")
    embeddings_list = []

    # Iterate over the blocks and extract the word embeddings for each one
    for block in blocks:
        # Tokenize the block into individual words
        words = block.split()
        # Create an empty array to store the embeddings for each word
        embeddings = np.zeros((len(words), len(model_dict[next(iter(model_dict))])))
        # Iterate over the words and get their embeddings
        for i, word in enumerate(words):
            if word in model_dict:
                embeddings[i] = model_dict[word]
        embeddings_list.append(embeddings)

    return embeddings
