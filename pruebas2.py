import torch
import os   
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset


# Definición del modelo CNN
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Definir dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet = self.data.iloc[idx, :-1]  # Excluye la última columna que es la etiqueta
        label = self.data.iloc[idx, -1]   # Obtiene la etiqueta (última columna)
        tweet_tensor = torch.tensor(tweet.values, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return tweet_tensor, label_tensor

# Función para entrenar el modelo
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        tweets, labels = batch
        predictions = model(tweets).squeeze(1)
        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Función para evaluar el modelo
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            tweets, labels = batch
            predictions = model(tweets).squeeze(1)
            loss = criterion(predictions, labels)
            acc = accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Función para calcular la precisión
def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


############# LOAD DATASET #############
os.chdir('C:/Users/Rafalete/Desktop/TFG/datasets/')
df = pd.read_csv('df_final.tsv', sep='\t', header=0)

df['label'] = df['label'].replace({'F': 0, 'T': 1})

y = df['label'] # variable de estudio

X = df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

######### DEEP LEARNING #########
# Dividir el dataset en entrenamiento y prueba
train_data, test_data = train_test_split(df, test_size=0.2)

# Crear instancias de los datasets personalizados
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

# Crear DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Obtener el tamaño del vocabulario (número de palabras únicas)
vocab_size = len(set(df.drop('label', axis=1).values.flatten()))

# Definir hiperparámetros
INPUT_DIM = vocab_size  
EMBEDDING_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

# Crear el modelo
model = CNN(INPUT_DIM, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# Definir la función de pérdida y el optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Entrenar el modelo
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')