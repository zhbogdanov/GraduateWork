import json
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ids, texts, labels):
        super(MyDataset, self).__init__()
        self.X = [texts[i] for i in ids]
        self.y = [labels[i] for i in ids]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class TERRaReader:
    def read(self, data_path: str):
        dataset = {}
        for filename in ['train.jsonl', 'val.jsonl']:
            dataset[filename.split('.')[0]] = self._build_data(data_path + '/' + filename)
        return dataset

    @staticmethod
    def _build_data(data_path):
        data = {}
        with open(data_path, 'r') as f:
            for line in f:
                jline = json.loads(line)
                if 'label' in jline:
                    data[jline['premise'], jline['hypothesis'].replace('ли ', '').replace('?', '')] = int(
                        jline['label'] == 'entailment')
        return list(data.items())


class DaNetQAReader:
    def read(self, data_path: str):
        dataset = {}
        for filename in ['train.jsonl', 'val.jsonl']:
            dataset[filename.split('.')[0]] = self._build_data(data_path + '/' + filename)
        return dataset

    @staticmethod
    def _build_data(data_path):
        data = {}
        with open(data_path, 'r') as f:
            for line in f:
                jline = json.loads(line)
                if 'label' in jline:
                    data[jline['passage'].lower(), jline['question'].lower()] = int(jline['label'] == True)
        return list(data.items())


class ParaphraseReader:
    def read(self, data_path: str):
        dataset = {}
        for filename in ['train.jsonl', 'test.jsonl']:
            dataset[filename.split('.')[0]] = self._build_data(data_path + '/' + filename)
        return dataset

    @staticmethod
    def _build_data(data_path):
        data = {}
        with open(data_path, 'r') as f:
            for line in f:
                jline = json.loads(line)
                if 'label' in jline:
                    data[jline['text1'], jline['text2']] = int(jline['label'] == True)
        return list(data.items())
