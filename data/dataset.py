from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, filename, logger):
        """
        Initialization data file path and other data-related configurations 
        Read data from data file
        Preprocess the data
        """
        pass
    def __len__(self):
        """
        Dataset length
        """
        pass
    def __getitem__(self, index):
        """
        Return a set of data pairs (data[index], label[index])
        """
        pass
    @staticmethod 
    def collate_fn(batch_data):
        """
        As parameters to torch.utils.data.DataLoader, Preprocess batch_data
        """
        pass
    def __read_data(self):
        pass
    def __preprocess_data(self):
        pass 
