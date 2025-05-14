# %% Import
import pandas as pd
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from transformers import AutoTokenizer, AutoModel


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

    # Load the model
model = AutoModel.from_pretrained(
    "bert-base-uncased"
)

# Set the model to evaluation mode
model = model.eval()
def load_data(focal_word, is_train, n_samples=100, random_state=42):
    data_type = "train" if is_train else "test"
    data_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.data.txt"
    label_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.gold.txt"

    data_table = pd.read_csv(
        data_file,
        sep="\t",
        header=None,
        dtype={"word_pos": int, "sentence": str},
        names=["word_pos", "sentence"],
    )
    label_table = pd.read_csv(
        label_file,
        sep="\t",
        header=None,
        dtype={"label": int},
        names=["label"],
    )
    combined_table = pd.concat([data_table, label_table], axis=1)
    return combined_table.sample(n_samples, random_state=random_state)

def train_classifier(emb, labels):
    """
    Train a classifier on the embeddings.
    """
    clf = LinearDiscriminantAnalysis(n_components=1)
    clf.fit(emb, labels)
    return clf

def evaluate_classifier(clf, emb, labels):
    """
    Evaluate the performance of the classifier.
    """
    prediction = clf.predict(emb)
    accuracy = np.mean(prediction == labels)
    return accuracy


# %% Test -----------

def load_data(focal_word, is_train, n_samples=100):
    data_type = "train" if is_train else "test"
    data_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.data.txt"
    label_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.gold.txt"

    data_table = pd.read_csv(
        data_file,
        sep="\t",
        header=None,
        dtype={"word_pos": int, "sentence": str},
        names=["word_pos", "sentence"],
    )
    label_table = pd.read_csv(
        label_file,
        sep="\t",
        header=None,
        dtype={"label": int},
        names=["label"],
    )
    combined_table = pd.concat([data_table, label_table], axis=1)
    return combined_table.sample(n_samples)

focal_word = "java"
train_data = load_data(focal_word, is_train=True)
test_data = load_data(focal_word, is_train=False)

with torch.no_grad():
    # Get the embeddings of the test set
    emb_train = get_token_embedding(train_data["sentence"].values.tolist(), train_data["word_pos"].values, model, tokenizer)

    # Get the embeddings of the test set
    emb_test = get_token_embedding(test_data["sentence"].values.tolist(), test_data["word_pos"].values, model, tokenizer)

# Train the classifier
clf = train_classifier(emb_train, train_data["label"].values)

# Get the predictions
prediction = clf.predict(emb_test)

# Evaluate the performance
accuracy = evaluate_classifier(clf, emb_test, test_data["label"].values)


assert accuracy > 0.91, f"Accuracy is {accuracy}, which is not greater than 0.91"
