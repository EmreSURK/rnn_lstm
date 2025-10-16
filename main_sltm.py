import os
import numpy as np
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re


import sentence_data
data = sentence_data.data

# PyTorch için random seed
torch.manual_seed(42)
np.random.seed(42)


# Hiperparametreler - İYİLEŞTİRİLMİŞ
VOCAB_SIZE = 5000
MAX_LEN = 15  
EMBEDDING_DIM = 64 
RNN_UNITS = 64 
BATCH_SIZE = 16  
EPOCHS = 50  
RANDOM_STATE = 42
LEARNING_RATE = 0.001
DROPOUT = 0.3 


class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class ImprovedLSTMModel(nn.Module):
    """İyileştirilmiş LSTM Modeli - RNN yerine LSTM ve Bidirectional"""
    def __init__(self, vocab_size, embedding_dim, rnn_units, max_len, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM (hem ileri hem geri okur)
        self.lstm = nn.LSTM(
            embedding_dim, 
            rnn_units, 
            batch_first=True, 
            bidirectional=True,  # Bidirectional eklendi
            num_layers=2,  # 2 katmanlı LSTM
            dropout=dropout if dropout > 0 else 0
        )
        
        # Dropout katmanı (overfitting'i önler). Nöronları rastgele kapatır. 
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional olduğu için output_size * 2.
        # 128'den düşürüyoruz, rrn'e göre daha büyük, bu yüzden iki adım.
        self.fc1 = nn.Linear(rnn_units * 2, 32)
        self.fc2 = nn.Linear(32, 1)

        # pozitif değerler aynen geçer, negatif değerler 0'a döner.
        # sonra sigmoid fonksiyonu ile 0-1 arasına çeker.
        # hızlı hesaplama, gradient vanishing problemi bulunmaz.
        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM çıktısı
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Son zaman adımını al
        last_output = lstm_out[:, -1, :]
        
        
        # Fully connected katmanlar
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)


def create_vocab(texts: List[str]) -> dict:
    """Kelime dağarcığı oluştur"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
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
    texts = [t for t, _ in dataset]
    labels = [y for _, y in dataset]
    
    vocab = create_vocab(texts)
    sequences = [text_to_sequence(text, vocab) for text in texts]
    padded_sequences = pad_sequences(sequences, MAX_LEN)
    
    return padded_sequences, labels, vocab


def train_model(model, train_loader, val_loader, epochs, device):
    """Modeli eğit"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (öğrenme hızını otomatik ayarlar)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, verbose=True
    # )
    
    
    for epoch in range(epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            
            # Gradient clipping (exploding gradient'i önler)
            # max gradient'i 1 e eşitlemek için çarpanı hesplar, tüm gradientleri bu çarpan ile çarpar. 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validasyon
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        # Learning rate scheduling
        # scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    return model


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
    print("=" * 70)
    
    # Veri hazırlama
    sequences, labels, vocab = prepare_data(data)
    
    # Train/validation split
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
    model = ImprovedLSTMModel(
        vocab_size, EMBEDDING_DIM, RNN_UNITS, MAX_LEN, DROPOUT
    ).to(device)
    
    
    # Eğitim
    print("\n🚀 Eğitim başlıyor...")
    model = train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Validasyon seti üzerinde detaylı değerlendirme
    print("\n" + "=" * 70)
    print("📊 VALIDASYON SETİ DETAYLI DEĞERLENDİRME")
    print("=" * 70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            predicted = (outputs.squeeze() > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    
    # Örnek tahminler
    examples = [
        "bugün harika hissediyorum",
        "moralim çok bozuk", 
        "keyfim yerinde ve mutluyum",
        "biraz yorgunum ama keyfim yerinde",
        "çok kötü bir gün",
        "mutlu bir günümde değilim",
    ]
    
    print("\n" + "=" * 70)
    
    if any("bugün" in text for text, _ in data):
        print("bugün kelimesi var")
    else:
        print("bugün kelimesi yok")

    print("🎯 ÖRNEK TAHMİNLER")
    print("=" * 70)
    for text in examples:
        label, prob = predict_sentiment(model, vocab, text, device)
        sentiment = "Pozitif ✅" if label == 1 else "Negatif ❌"
        print(f"'{text}'")
        print(f"  → {sentiment} (olasılık: {prob:.3f})\n")
    
    print("=" * 70)
    print("✅ Eğitim tamamlandı!")

