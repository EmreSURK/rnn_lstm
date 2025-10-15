# 🤖 RNN Duygu Analizi - Türkçe

PyTorch kullanarak Türkçe duygu analizi yapan LSTM modeli.

## 🚀 Hızlı Başlangıç

### Geliştirilmiş Modeli Çalıştırın (Önerilen)
```bash
python3 main_improved.py
```

### Orijinal (Sorunlu) Modeli Çalıştırın
```bash
python3 main.py
```

## 📊 Model Karşılaştırması

| Özellik | Orijinal Model | İyileştirilmiş Model |
|---------|----------------|---------------------|
| **Accuracy** | %96.67 (ama yanlış!) | %98.96 ✅ |
| **Test Başarısı** | %20 | %100 ✅ |
| **Mimari** | Vanilla RNN | Bidirectional LSTM |
| **Parametre** | 6,913 | 179,457 |
| **Dropout** | ❌ | ✅ (0.3) |
| **Layers** | 1 | 2 |
| **LR Scheduling** | ❌ | ✅ |

## 📖 Detaylı Analiz

Sorunların detaylı analizi ve çözümleri için [SONUC_RAPORU.md](SONUC_RAPORU.md) dosyasına bakın.

## 🎯 Örnek Kullanım

Model eğitildikten sonra tahmin yapmak için:

```python
from main_improved import ImprovedLSTMModel, predict_sentiment, prepare_data
import torch

# Modeli yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedLSTMModel(vocab_size, 64, 64, 20, 0.3).to(device)
model.load_state_dict(torch.load('best_model.pth'))

# Tahmin yap
text = "bugün çok mutluyum"
label, prob = predict_sentiment(model, vocab, text, device)
print(f"Duygu: {'Pozitif' if label == 1 else 'Negatif'} ({prob:.2%})")
```

## 📦 Gereksinimler

```bash
pip install torch numpy scikit-learn
```

## 🔍 Ne Yanlıştı?

**Ana Sorunlar:**
1. ❌ Kelime frekans çarpıklığı (örn: "bugün" 128 kez negatif, 57 kez pozitif)
2. ❌ Vanilla RNN - bağlam kaybı
3. ❌ Küçük model kapasitesi
4. ❌ Regularization eksikliği
5. ❌ Overfitting

## ✅ Nasıl Düzeltildi?

**Çözümler:**
1. ✅ LSTM (Long Short-Term Memory) kullanıldı
2. ✅ Bidirectional yapı eklendi (ileri + geri)
3. ✅ 2 katmanlı derin ağ
4. ✅ Dropout (%30) eklendi
5. ✅ Gradient clipping
6. ✅ Learning rate scheduling
7. ✅ Daha büyük embedding (32→64)
8. ✅ Daha büyük hidden units (32→64)

## 🎓 Öğrenilenler

1. **Yüksek validation accuracy ≠ İyi model**
   - Gerçek test örnekleriyle mutlaka kontrol edin!

2. **Kelime frekans analizi kritik**
   - Veri dengesizliklerini tespit edin

3. **LSTM > RNN**
   - Bağlam önemli olan görevler için LSTM kullanın

4. **Regularization şart**
   - Dropout, gradient clipping kullanın

## 📂 Dosya Yapısı

```
RNN_duygu_tahmin/
├── main.py              # ❌ Orijinal (sorunlu) model
├── main_improved.py     # ✅ İyileştirilmiş model
├── best_model.pth       # 💾 Eğitilmiş model
├── SONUC_RAPORU.md      # 📊 Detaylı analiz raporu
└── README.md            # 📄 Bu dosya
```

## 🎯 Test Sonuçları

```
'bugün harika hissediyorum'  → Pozitif ✅ (99.1%)
'moralim çok bozuk'          → Negatif ✅ (1.6%)
'keyfim yerinde ve mutluyum' → Pozitif ✅ (98.9%)
'çok kötü bir gün'           → Negatif ✅ (0.4%)
'berbat hissediyorum'        → Negatif ✅ (1.1%)
'çok mutluyum bugün'         → Pozitif ✅ (99.3%)
'üzgün ve yorgunum'          → Negatif ✅ (0.4%)
'pozitif enerji doluyum'     → Pozitif ✅ (99.2%)
```

**Başarı Oranı: %100** 🎉

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not:** Detaylı teknik analiz için [SONUC_RAPORU.md](SONUC_RAPORU.md) dosyasını okuyun.


