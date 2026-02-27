# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement 
To develop and implement a Recurrent Neural Network (RNN) model capable of capturing temporal dependencies in historical stock closing prices and forecasting future stock values based on previously observed trends.

## Dataset
The dataset includes historical stock market records containing daily closing prices of a chosen company. Prior to training and evaluation, the data is preprocessed using normalization techniques and transformed into sequential input samples suitable for the RNN model.

## DESIGN STEPS

## STEP 1:
Acquire historical stock price data for the selected company and perform preprocessing steps such as handling missing values, normalization, and generating time-series sequences.

## STEP 2:
Divide the processed dataset into training and testing subsets and transform the sequences into tensors, organizing them using appropriate data loaders for efficient training.

## STEP 3:
Construct a Recurrent Neural Network (RNN) architecture with suitable input size, hidden layers, and output layer to capture temporal patterns in stock price movements.

## STEP 4:
Select an appropriate loss function such as Mean Squared Error (MSE) and choose an optimizer like Adam to guide the learning process.

## STEP 5:
Train the RNN model across several epochs by feeding sequential data, computing loss, and updating model parameters through backpropagation through time.

## STEP 6:
Test the trained model on unseen data and evaluate its performance by visualizing training loss and comparing predicted stock prices with actual market values.

## PROGRAM

### Name: SWETHA K

### Register Number: 212224230284

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')
df_train.head()
df_test.head()

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc=nn.Linear(hidden_size, output_size)
    def forward(self,x):
      out,_=self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion =nn.MSELoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f'Epoch [{epoch+1}/{epochs}],Loss: {total_loss/len(train_loader):.4f}')
# Plot training loss
  print('Name:  SWETHA K')
  print('Register Number: 212224230284  ')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model(model,train_loader,criterion,optimizer)

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:   SWETHA K ')
print('Register Number: 212224230284 ')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="1110" height="636" alt="image" src="https://github.com/user-attachments/assets/9187c01a-2049-4a3e-93eb-52c03c59dce8" />

## True Stock Price, Predicted Stock Price vs time

<img width="1222" height="738" alt="image" src="https://github.com/user-attachments/assets/9996d3bd-09d2-48ed-9a70-04712b28ea3d" />


### Predictions

<img width="616" height="76" alt="image" src="https://github.com/user-attachments/assets/997e5179-dcee-4045-9976-2bf66d227d44" />


## RESULT

Historical stock price data was preprocessed and used to train an RNN model, which successfully minimized training loss and accurately predicted stock price trends as shown by the close alignment between actual and predicted values.


