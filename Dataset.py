from torch.utils.data import Dataset
import pickle as pk


class NewsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_data():
    embedding_file = open("./_74429_V01_EMs/embeddings.pkl", "rb")
    embedding_label_file = open("./_74429_V01_EMs/embedding_classes.pkl", "rb")
    embeddings = pk.load(embedding_file)["embeddings"]
    classes = pk.load(embedding_label_file)["embedding_classes"]
    embedding_file.close()
    embedding_label_file.close()
    return embeddings, classes
