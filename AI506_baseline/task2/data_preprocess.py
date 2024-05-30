from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import glob

class Task2_Data(Dataset):
    def __init__(self, root, train_type="train"):
        f = open(os.path.join(root, "task2_data.txt"), 'r')
        skip = True
        order_dict = {}
        while True:
            line = f.readline()
            if skip:
                skip = False
                continue

            if not line: break

            data_list = line.split(",")
            order_number = int(data_list[0])
            if order_number not in order_dict:
                order_dict[order_number] = [(int(data_list[1]), int(data_list[2]), int(data_list[3]), int(data_list[4]), int(data_list[5]))] 
            elif order_number in order_dict:
                order_dict[order_number].append((int(data_list[1]), int(data_list[2]), int(data_list[3]), int(data_list[4]), int(data_list[5])))
        f.close()

        if train_type == "train":
            f_data = open(os.path.join(root, "task2_train_label.txt"), 'r')
        else:
            f_data = open(os.path.join(root, "task2_valid_label.txt"), 'r')

        order = []
        data = []
        label = []
        while True:
            line = f_data.readline()
            if not line: break
            data_list = line.split("\t")
            order.append(int(data_list[0]))
            data.append(int(data_list[1]))
            label.append(int(data_list[2]))
        f_data.close()


        self.order_dict = order_dict
        self.order = order
        self.data = data 
        self.label = label

        assert len(self.order) == len(self.data)
        assert len(self.data) == len(self.label)

    def __getitem__(self, index):
        order_number = self.order[index]
        data_number = self.data[index]
        label = self.label[index]

        total_product = 0
        max_len = 46
        product = []
        customer = []
        color = []
        size = []
        group = []
        labels = []
        products = self.order_dict[order_number]
        for data in products:
            if data[0] == data_number:
              product.append(data[0])
              customer.append(data[1])
              color.append(data[2])
              size.append(data[3])
              group.append(data[4])
              labels.append(label)
            else:
              product.append(data[0])
              customer.append(data[1])
              color.append(data[2])
              size.append(data[3])
              group.append(data[4])
              labels.append(-100)

            total_product += 1
        
        for _ in range(max_len - total_product):
            product.append(0)
            customer.append(0)
            color.append(0)
            size.append(0)
            group.append(0)
            labels.append(-100)

        return torch.Tensor(product).long(), torch.Tensor(customer).long(), torch.Tensor(color).long(), torch.Tensor(size).long(), torch.Tensor(group).long(), torch.Tensor(labels).long()

    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    data = Task2_Data("./", train_type="train")
    print(len(data))
    a = DataLoader(data, batch_size=1, shuffle=False)
    for i, k in enumerate(a):
        print(k[0], k[1], k[2], k[3], k[4], k[5],  sep="\n")
        break