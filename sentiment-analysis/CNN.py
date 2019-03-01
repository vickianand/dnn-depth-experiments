import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
nlp = spacy.load('en')

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device)

def predict_sentiment(sentence, min_len=5):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
#         super().__init__()
        
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0],embedding_dim))
#         self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1],embedding_dim))
#         self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2],embedding_dim))
#         self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
        
#         #x = [sent len, batch size]
        
#         x = x.permute(1, 0)
                
#         #x = [batch size, sent len]
        
#         embedded = self.embedding(x)
                
#         #embedded = [batch size, sent len, emb dim]
        
#         embedded = embedded.unsqueeze(1)
        
#         #embedded = [batch size, 1, sent len, emb dim]
        
#         conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
#         conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
#         conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
#         #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
#         pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
#         pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
#         pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
#         #pooled_n = [batch size, n_filters]
        
#         cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

#         #cat = [batch size, n_filters * len(filter_sizes)]
            
#         return self.fc(cat)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [sent len, batch size]       
        x = x.permute(1, 0)
        #x = [batch size, sent len]
        embedded = self.embedding(x)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 150
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    #returns fraction
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print('Epoch: ', epoch+1, 'Train Loss: ', train_loss, ' Train Acc: ',train_acc*100 , 'Val. Loss: ', valid_loss, 'Val. Acc: ', valid_acc*100)


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print('Test Loss: ', test_loss, 'Test Acc: ', test_acc*100)



