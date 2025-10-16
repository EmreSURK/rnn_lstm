## Gradient/Parametre Normu Raporu

### Tanımlar
- **param_norm (‖θ‖2)**: Model ağırlıklarının L2 normu (mevcut büyüklük).
- **grad_norm (‖∇θL‖2)**: O adımda hesaplanan gradyan vektörünün L2 normu (güncelleme sinyali).
- Öğrenme oranı: η = 0.001
- Yaklaşık güncelleme büyüklüğü: ‖Δθ‖ ≈ η · ‖∇θL‖
- Oransal etki: ‖Δθ‖ / ‖θ‖

---

### Normal eğitim örneği (stabil)
Loglardan örnek değerler:
- grad_norm ≈ 0.33, param_norm ≈ 69.77

Hesap:
- ‖Δθ‖ ≈ 0.001 × 0.33 = 0.00033
- Oran: 0.00033 / 69.77 ≈ 4.7e-6 (milyonda birkaç)

Yorum:
- Güncellemeler parametre büyüklüğüne kıyasla çok küçük; ağırlıklar şişmiyor.
- Eğitimde loss/acc stabil seyreder, NaN/Inf görülmez.

Mini-batch varyasyonu örneği (normal oynaklık):
- grad_norm: 0.136 → 0.335 → 0.026 (hepsi 1’in altında)
- param_norm: ≈ 69.77 (hemen hemen sabit)

---

### Exploding gradient örneği (ani sıçrama/NaN)
Verilen değerler (örnek):
- grad_norm = 1783.42, param_norm = 965.10
- (Logda loss: NaN)

Hesap:
- ‖Δθ‖ ≈ 0.001 × 1783.42 = 1.78342
- Oran: 1.78342 / 965.10 ≈ 0.00185 ≈ %0.185/step

Yorum:
- Tek adımda anlamlı bir ağırlık değişimi; ardışık adımlarda parametreler hızla büyüyebilir.
- Loss’un NaN olması sayısal istikrarsızlığı doğrular; bu örüntü exploding’e tipiktir.

---

### Hızlı karşılaştırma
- Stabil durumda: grad_norm ≪ param_norm ve η küçük → ‖Δθ‖ / ‖θ‖ çok küçük.
- Exploding durumda: grad_norm çok büyük (10²–10⁴, hatta Inf) → ‖Δθ‖ büyük, loss/acc kaotik/NaN.
- Vanishing durumda: grad_norm kronik çok küçük (~1e-4 ve altı), loss ~0.693 plato, acc ~0.5.

---

### Pratik notlar
- İzleme: her N adımda `grad_norm` ve `param_norm` kaydı, hareketli ortalama ile anomali tespiti.
- Koruma: `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)` ile patlamaları önleme.
- Aktivasyon kontrolleri: RNN gizil durum normları (h_t) ve doygunluk (sigmoid/tanh) gözlemi.

---

### Vanishing gradient örneği (öğrenme durgun, 0.69 civarı plato)
Örnek belirtiler:
- Loss ~0.693 civarında uzun süre plato, doğruluk ~0.5 civarı.
- grad_norm kronik çok küçük (≈1e-4 ve altı), param_norm normal aralıkta.

Varsayımsal değerler ve hesap (η = 0.001):
- grad_norm = 0.0008, param_norm = 3.12
- ‖Δθ‖ ≈ 0.001 × 0.0008 = 8e-7
- Oran: 8e-7 / 3.12 ≈ 2.6e-7 (son derece küçük)

Yorum:
- Güncellemeler neredeyse etkisiz kaldığından öğrenme ilerlemez.
- Çözüm yolları: daha derin/uzun bağımlılıklarda LSTM/GRU, daha iyi başlatma, normalizasyon, uygun aktivasyonlar, LR/optimizasyon ayarı ve gerektiğinde curriculum/önişleme.
