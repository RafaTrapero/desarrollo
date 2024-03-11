import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

os.chdir('C:/Users/Rafalete/Desktop/TFG/datasets/')
df = pd.read_csv('df_final.tsv', sep='\t', header=0)

######### DEEP LEARNING #########

df['label'] = df['label'].replace({'F': 0, 'T': 1})
df.to_csv('df_pruebas.csv')
print('dataset',df)

y = df['label'] # variable de estudio

X = df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Convertir tus datos a tensores de tipo float en lugar de long para trabajar con tokens
train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.long), torch.tensor(y_train.values, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long))

print('train_dataset', train_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        input_data, labels = batch
        optimizer.zero_grad()
        outputs = model(input_data, labels=labels)  # Ajustar la llamada al modelo con los datos de entrada y etiquetas
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.eval()
total_correct = 0

for batch in test_dataloader:
    input_data, labels = batch
    with torch.no_grad():
        outputs = model(input_data=input_data)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    total_correct += torch.sum(predictions == labels).item()

accuracy = total_correct / len(y_test)
print(f'Accuracy on test set: {accuracy}')
