import torch
import matplotlib.pyplot as plt
import os
from einops import rearrange

class MotorImgDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, dir, window_size, time_dim=736, normalize=True):
            super().__init__()
            self.dir = dir
            self.window_size = window_size
            self.time_dim = time_dim
            #number by which total number of data is multiplied because of windowing
            self.mult_data = self.time_dim-self.window_size
            # data for normalization
            self.normalize = normalize
            self.mean, self.std = torch.load("E:/Documents/datasets/MotorImgTorch_mean_std.pt")

        def __len__(self):
            return self.mult_data*len(os.listdir(self.dir))//2

        def __getitem__(self, idx):
            file_num = idx//(self.mult_data)
            data_path = os.path.join(self.dir,str(file_num))
            x = torch.load(f"{data_path}X.pt")
            window = x[idx%self.mult_data:idx%self.mult_data+self.window_size]
            window = rearrange(window,"h w c -> c h w")
            label = torch.load(f"{data_path}Y.pt").type(torch.int64)

            # normalization
            if self.normalize:
                window = (window-self.mean)/self.std

            return window, label

    def get_train_val_test_dataloader(self, train_test_val_percentage, batch_size, vbatch_size = 1, tbatch_size = 1, window_size = 160, time_dim=736, normalize=True, num_workers=0, pin_memory=True):
        assert len(train_test_val_percentage) == 3, "train_test_val_percentage should be of length 3"
        assert sum(train_test_val_percentage) == 1, "train_test_val_percentage should sum to 1"
        
        dataset = self.Dataset(self.data_path, window_size, time_dim=time_dim, normalize=normalize)
        dataset_num = len(dataset)
        test_num = int(dataset_num*train_test_val_percentage[1])
        val_num = int(dataset_num*train_test_val_percentage[2])
        train_num = dataset_num-val_num-test_num
        print(f"train_num: {train_num}, val_num: {val_num}, test_num: {test_num}")
        train_data,val_data,test_data = torch.utils.data.random_split(dataset,[train_num,val_num,test_num])
        training_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(val_data, batch_size=vbatch_size, shuffle=True)
        testing_loader = torch.utils.data.DataLoader(test_data, batch_size=tbatch_size, shuffle=True)
        return training_loader, validation_loader, testing_loader

    def get_dataset(self, window_size = 160, time_dim=736, normalize=True):
        return self.Dataset(self.data_path, window_size, time_dim=time_dim, normalize=normalize)

if __name__=="__main__":
    pt_file_path = "E:/Documents/datasets/MotorIMGTorch"
    print("some test data:")
    test_data = torch.load(os.path.join(pt_file_path,"1X.pt"))
    print(test_data.shape)
    test_idx = 0
    eeg_dataset = MotorImgDataset.Dataset(pt_file_path,160)
    plt.imshow(eeg_dataset[test_idx][0][0])
    print(f"label: {eeg_dataset[test_idx][1]}")
    plt.show()