
import os
import numpy as np
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import pandas as pd

# import sentence_data
# data = sentence_data.data

import clean_sentence_data # (bugün,çok kaldırıldı)
data = clean_sentence_data.data


"""tüm random değerlerde aynı seed'den üretilsin, her seferinde aynı olsun. """
RANDOM_STATE = 42


"""max kelime(token) sayısı. 
token: kelime ya da karakter ya da kelime fragmenti. geliyor-um yapısı da bir token.
bu sayıya sığmayanlar OOV (Out-Of-Vocabulary) olarak kabul edilir 
ve UNK(unknown) token'ına eşitlenir. 
Genelde OOV oranı %10 un altında tutulur. 
PAD ve UNK token'ları da dahil edilir.
"""
VOCAB_SIZE = 5000 


"""
her satırdaki max kelime sayısı. boşluklar PAD ile doldurulur.
PAD id'si 0'dır.
[mutlu_id, değilim_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
BOS EOF -> beginning of sequence, end of sequence.
"""
MAX_LEN = 15  


"""
embedding boyutu. her kelimeyi bir vektörle temsil eder.
"mutlu" → [0.1, -0.3, 0.7, ..., 0.2] (32 sayı).
Değerler başta rastgele seçilir.
eğitim sırasında, her bir epoch'ta: tahmin -> hata hesabı(loss) -> gradyan hesabı(gradient) -> embedding güncelleme.
5000 x 32 = 160,000 -> 160k parametreli bir model.
"""
EMBEDDING_DIM = 32  


"""
hidden layerdaki nöron sayısı.
hidden layerdaki nöron sayısı ne kadar büyükse, model o kadar iyi öğrenebilir.
"""
RNN_UNITS = 32  

"""
aynı anda kaç örnek işlensin?
büyük olursa daha çok bellek kullanır.
gradyanlar ortalaması alınır. böylece daha stabil bir gradyan elde edilir.
"""
BATCH_SIZE = 8  

EPOCHS = 50  


"""
gradyan ile çarpılır.
büyük değerler daha hızlı öğrenmesini sağlar ama beraberinde hatalar riskler getirebilir. (gradient exploding, overfitting, etc.)
"""
LEARNING_RATE = 0.001  

# PyTorch için random seed. rastgele seçilen değerleri her run edişte aynı seçsin. 
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, max_len):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        """
        num_layers: RNN katman sayısı (hidden layer sayısı). 
          az olursa model daha az öğrenebilir,
          çok olursa overfitting veya gradient vanishing problemi olabilir. 
          hidden layer arttıkça gradient küçülür. 
          LSTM de gradient problemi daha az
        """
        self.rnn = nn.RNN(embedding_dim, rnn_units, num_layers=1, batch_first=True)

        """
        tam bağlı katman.
        1 çıktı, 0 ya da 1
        [0.1, 0.3, ..., 0.7] → [W×x + b] → [0.8]
        Bu aşamada bias varsayılan olarak eklenir.
        farklı kullanımlar mevcut:
          self.fc = nn.Linear(rnn_units, 16)
          self.fc = nn.Linear(16, 1)
        """
        self.fc = nn.Linear(rnn_units, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        # Son zaman adımını al
        last_output = rnn_out[:, -1, :]
        output = self.fc(last_output)
        return self.sigmoid(output)


def create_vocab(texts: List[str]) -> dict:
    """Kelime dağarcığı oluştur"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    """
    mutlu : 4
    kötü : 8
    """
    word_count = {}
    
    for text in texts:
        words = text.lower().split()
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
    
    # En sık kullanılan kelimeleri al
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for i, (word, count) in enumerate(sorted_words[:VOCAB_SIZE-2]):
        vocab[word] = i + 2
    
    return vocab


def text_to_sequence(text: str, vocab: dict) -> List[int]:
    """Metni sayı dizisine çevir"""
    words = text.lower().split()
    sequence = []
    for word in words:
        sequence.append(vocab.get(word, vocab["<UNK>"]))
    return sequence


def pad_sequences(sequences: List[List[int]], max_len: int) -> List[List[int]]:
    """Dizileri sabit uzunluğa getir"""
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [0] * (max_len - len(seq)))
    return padded


def prepare_data(dataset: list) -> Tuple[List[List[int]], List[int], dict]:
    """Veriyi hazırla"""
    texts = [t for t, _ in dataset] # metni al 
    labels = [y for _, y in dataset] # labeli(sayısal değerleri) al
    
    vocab = create_vocab(texts)
    sequences = [text_to_sequence(text, vocab) for text in texts]
    padded_sequences = pad_sequences(sequences, MAX_LEN)
    
    return padded_sequences, labels, vocab


def train_model(model, train_loader, val_loader, epochs, device):
    """Modeli eğit"""
    """
    Binary Cross Entropy. model tahmini ile gerçek değer arasındaki farkı hesaplar.
    Loss = -[y*log(p) + (1-y)*log(1-p)]
    p:model tahmini, y:gerçek değer
    
    y_true = 1  # Pozitif
    # Model tahmini(p):
    y_pred = 0.8  # %80 pozitif

    # Loss hesaplama:
    loss = -[1*log(0.8) + (1-1)*log(1-0.8)]
     = -[log(0.8) + 0]
     = -log(0.8)
     = 0.223
    """
    criterion = nn.BCELoss()

    """
    Adaptive Moment Estimation
    Eğitim sırasında önceki gradyanları hatırlar.
    Parametreleri optimize eder.
    loss.backward() tarafından hesaplanan gradyanları(grad1) kullanacak.
    
    v1 = β1 * v0 + (1-β1) * grad1  # Momentum
    s1 = β2 * s0 + (1-β2) * grad1² # Adaptive learning rate
    param1 = param0 - lr * v1 / √s1

    v1 - yeni momentum 
    b1 - geçmiş momentum değeri(sabit) 
    v0 - önceki momentum değeri 
    grad1 - mevcut gradyan 
    
    s1 - yeni squared gradient 
    b2 - geçmiş squared gradient değeri(sabit) 
    s0 - önceki squared gradient değeri 
    grad1² - mevcut gradyanın karesi 
    """

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

    """ eğer patience sayısına ulaştığında hala öğrenmede ilerleme yoksa eğitimi durdur. """
    best_val_loss = float('inf')

    patience = 10 
    patience_counter = 0
    
    for epoch in range(epochs):
        # Eğitim moduna geç 
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # backpropagation (hata azaltma) için.
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            #önceki gradientleri sıfırla.
            optimizer.zero_grad()

            #model tahmin yapıyor. (forward pass)
            # [[0.8], [0.2], [0.9], [0.1], ...]
            outputs = model(sequences)

            #squeeze -> [[0.8], [0.2], [0.9], [0.1], ...] -> [0.8, 0.2, 0.9, 0.1, ...]
            #hata hesabı. (loss function)
            loss = criterion(outputs.squeeze(), labels)

            #gradyan hesabı. (backward pass, backpropagation)
            loss.backward()

            # tüm parametreleri Adam algoritması ile güncelle. 
            optimizer.step()
            
            train_loss += loss.item() # .item() fonksiyonu tensor'u float'a çevirir.

            #batch'lerin cevaplarını topla, sınıflandır. 
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += labels.size(0)

            #tahmin doğru mu? doğruysa 1, yanlışsa 0.
            train_correct += (predicted == labels).sum().item()
        
        # gradient ve parameter normunu hesapla
        with torch.no_grad():
            grad_sq_sum = 0.0
            param_sq_sum = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq_sum += p.grad.data.pow(2).sum().item()
                param_sq_sum += p.data.pow(2).sum().item()
            grad_norm = grad_sq_sum ** 0.5
            param_norm = param_sq_sum ** 0.5
        print(f"grad_norm={grad_norm:.4f} param_norm={param_norm:.4f}")
        
        # Validasyon için test moduna geç.
        # güncelleme devredışı, dropout iptal, tüm nöronlar aktif, gradient hesabı devredışı. 
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader: # bu sefer validation seti üzerinden test yapıyoruz.
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


def predict_sentiment(model, vocab: dict, text: str, device) -> Tuple[int, float]:
    """Duygu tahmini yap"""
    model.eval()
    sequence = text_to_sequence(text, vocab)
    padded = pad_sequences([sequence], MAX_LEN)
    tensor = torch.LongTensor(padded).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prob = float(output.squeeze().item())
        label = 1 if prob >= 0.5 else 0
    
    return label, prob


if __name__ == "__main__":
    # Device belirleme
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri hazırlama
    sequences, labels, vocab = prepare_data(data)

    print(pd.DataFrame(sequences).head())
    print(pd.DataFrame(labels).head())
    print(pd.DataFrame(vocab.items()).head(20))
    
    # Train/validation verisini ayırıyoruz. 
    # x -> text, y -> label, f(x)=y
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # DataLoader oluştur
    train_dataset = SentimentDataset(X_train, y_train)
    val_dataset = SentimentDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model kurulum
    vocab_size = len(vocab)
    model = SimpleRNNModel(vocab_size, EMBEDDING_DIM, RNN_UNITS, MAX_LEN).to(device)
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Kelime dağarcığı boyutu: {vocab_size}")
    
    # Eğitim
    train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Örnek tahminler
    examples = [
        "bugün harika hissediyorum",
        "moralim çok bozuk", 
        "keyfim yerinde ve mutluyum",
        "biraz yorgunum ama keyfim yerinde",
        "çok kötü bir gün",
        "mutlu bir günümde değilim"
    ]
    
    if any("bugün" in text for text, _ in data):
        print("bugün kelimesi var")
    else:
        print("bugün kelimesi yok")

    print("\n=== Örnek Tahminler ===")
    for text in examples:
        label, prob = predict_sentiment(model, vocab, text, device)
        sentiment = "Pozitif ✅" if label == 1 else "Negatif ❌"
        print(f"'{text}'")
        print(f"  → {sentiment} (olasılık: {prob:.3f})\n")

