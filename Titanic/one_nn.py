# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import torch
import pandas as pd
from torch import nn

train_data = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")

features = ["Pclass", "Sex", "SibSp", "Parch"]

# y_train を別変数で定義
y_train = torch.tensor(train_data["Survived"].values, dtype=torch.float)

# X_train をTensorに変換
X_train = torch.tensor(pd.get_dummies(train_data[features]).astype(float).values, dtype=torch.float)

# X_test も同様に前処理
X_test = torch.tensor(pd.get_dummies(test_data[features]).astype(float).values, dtype=torch.float)

# モデル定義（線形回帰モデルは入力次元に合わせる）
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()

torch.manual_seed(42)
model_0 = LinearRegressionModel(X_train.shape[1])

epochs = 100
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# predictionsを出す
model_0.eval()
with torch.no_grad():
    predictions = (model_0(X_test) > 0.5).int().numpy()

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
