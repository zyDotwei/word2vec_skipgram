import torch.utils.data as Data


class BuildDataSet(Data.Dataset):

    def __init__(self, data_path, lens):
        self.data_path = data_path
        self.lens = lens
        self.pf = open(self.data_path, 'r', encoding='utf-8')

    def __getitem__(self, item):
        data = self.pf.__next__()
        data = data.strip().split(' ')
        print(item)
        if item == (self.lens - 1):
            self.pf.close()
            self.pf = open(self.data_path, 'r', encoding='utf-8')
        return int(data[1]), int(data[3])

    def __len__(self):
        return self.lens

    def __del__(self):
        if self.pf is not None:
            self.pf.close()


# if __name__ == '__main__':
#     trainDataSet = BuildDataSet('../temp/temp_all_5_skipgram.txt', 1088)
#     loader = Data.DataLoader(dataset=trainDataSet, batch_size=3)
#
#     for (trains, labels) in loader:
#         print(trains, labels)
