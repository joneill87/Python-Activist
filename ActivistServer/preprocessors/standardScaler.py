from dataset import Dataset
from sklearn import preprocessing


class StandardScaler:

    def process_step(self, data: Dataset) -> Dataset:
        scaler = preprocessing.StandardScaler().fit(data.features)
        if data.labelled.row_count > 0:
            data.labelled.features = scaler.transform(data.labelled.features)

        if data.unlabelled.row_count > 0:
            data.unlabelled.features = scaler.transform(data.unlabelled.features)

        return data
