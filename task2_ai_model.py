

import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from transformers import BertModel
from pathlib import Path
# 调调参
batch_size = 32

text_max_length = 128

epochs = 100

learn_rate = 3e-5

validation_ratio = 0.15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


log_per_step = 100


dataset_dir = Path("./data/data231041")
os.makedirs(dataset_dir) if not os.path.exists(dataset_dir) else ''


model_dir = Path("./model/bert_checkpoints")

os.makedirs(model_dir) if not os.path.exists(model_dir) else ''

print("Device:", device)


pd_train_data = pd.read_csv(f'./{dataset_dir}/train.csv')
pd_train_data['title'] = pd_train_data['title'].fillna('')
pd_train_data['abstract'] = pd_train_data['abstract'].fillna('')

test_data = pd.read_csv(f'./{dataset_dir}/testB.csv')
test_data['title'] = test_data['title'].fillna('')
test_data['abstract'] = test_data['abstract'].fillna('')

pd_train_data['text'] = pd_train_data['title'].fillna('') + ' ' + pd_train_data['abstract'].fillna('') + ' ' + pd_train_data['Keywords'].fillna('')
test_data['text'] = test_data['title'].fillna('') + ' ' + test_data['abstract'].fillna('') + ' ' + pd_train_data['Keywords'].fillna('')


validation_data = pd_train_data.sample(frac=validation_ratio)
train_data = pd_train_data[~pd_train_data.index.isin(validation_data.index)]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    text, label = zip(*batch)
    text, label = list(text), list(label)

    src = tokenizer(text, padding='max_length',
                    max_length=text_max_length, return_tensors='pt', truncation=True,)

    return src, torch.LongTensor(label)


class MyDataset(Dataset):

    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode

        if mode == 'train':
            self.dataset = train_data
        elif mode == 'validation':
            self.dataset = validation_data
        elif mode == 'test':

            self.dataset = test_data
        else:
            raise Exception("Unknown mode {}".format(mode))

    def __getitem__(self, index):

        data = self.dataset.iloc[index]

        text = data['text']

        if self.mode == 'test':

            label = data['uuid']
        else:
            label = data['label']

        return text, label

    def __len__(self):
        return len(self.dataset)


train_dataset = MyDataset('train')
validation_dataset = MyDataset('validation')

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

inputs, targets = next(iter(train_loader))
print("inputs:", inputs)
print("targets:", targets)


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.bert = BertModel.from_pretrained(
            'bert-base-uncased', mirror='tuna')

        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        return self.predictor(outputs)


model = MyModel()
model.to(device=device)

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


def to_device(dict_tensors):
    result = {}
    for k, v in dict_tensors.items():
        result[k] = v.to(device)
    return result


def validate():
    model.eval()
    total_loss = 0.
    total_correct = 0
    for inputs, targets in validation_loader:
        inputs, targets = to_device(inputs), targets.to(device)
        outputs = model(inputs)
        loss_value = loss(outputs.view(-1), targets.float())
        total_loss += float(loss_value)

        correct_num = (((outputs >= 0.5).float() *
                       1).flatten() == targets).sum()
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)


if(__name__ == "__main__"):
    model.train()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    total_loss = 0.

    step = 0


    best_accuracy = 0


    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):

            inputs, targets = to_device(inputs), targets.to(device)

            outputs = model(inputs)

            loss_v = loss(outputs.view(-1), targets.float())
            loss_v.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += float(loss_v)
            step += 1

            if step % log_per_step == 0:
                print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch +
                      1, epochs, i, len(train_loader), total_loss))
                total_loss = 0

            del inputs, targets

        accuracy, validation_loss = validate()
        print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(
            epoch+1, accuracy, validation_loss))
        # torch.save(model, model_dir / f"model_{epoch}.pt") 太吃硬盘寿命了，遭不住

        if accuracy > best_accuracy:
            torch.save(model, model_dir / f"model_best.pt")
            best_accuracy = accuracy
