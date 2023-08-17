from torch import nn
import pandas as pd
import torch
from task2_ai_model import MyModel, MyDataset, collate_fn
from torch.utils.data import DataLoader

model = torch.load(r'model\bert_checkpoints\model_best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = pd.read_csv(r'data\data231041\testB.csv')

model.eval()
dataset = MyDataset("test")
loader = DataLoader(dataset=dataset, batch_size=16,
                    shuffle=False, collate_fn=collate_fn)

results = []
for inputs, ids in loader:
    outputs = model(inputs.to(device))
    outputs = (outputs >= 0.5).int().flatten().tolist()
    ids = ids.tolist()
    results = results + [(id, result) for result, id in zip(outputs, ids)]


test_label = [pair[1] for pair in results]
test_data['label'] = test_label
test_data['Keywords'] = test_data['title'].fillna('')
test_data[['uuid', 'Keywords', 'label']].to_csv(
    'submit_task2.csv', index=None)  # type: ignore
