# Neural Network — ملخص شامل للكورس

> **Faculty of Computers and Information — Mansoura University**
> Neural Network Course

---

## نظرة عامة على الكورس

الكورس ده بيغطي **نظرية وتطبيقات الـ Neural Networks والـ Deep Learning** — من أساسيات الـ Machine Learning لحد أحدث المعماريات زي Transformers والـ Generative Models.

بنمر على كل مرحلة بالتفصيل:
- من أول ما نفهم **الفرق بين البرمجة التقليدية والتعلم الآلي**
- لحد ما نبني **شبكات عصبية معقدة** بتولّد صور وبتترجم لغات

الكورس مقسّم على **12 محاضرة رئيسية**، كل واحدة بتغطي جزء أساسي من عالم الـ Neural Networks.

---

## هيكل الملفات

| الملف | المحتوى | الصفحات |
|-------|---------|---------|
| `README.md` | الملخص الشامل ده — فيه نظرة عامة على كل المحاضرات | — |
| `Lecture_01_Introduction_Deep_Learning.md` | مقدمة + ML vs Programming + AI/ML/DL + History | 1–20 |
| `Lecture_02_Neural_Network_Basic.md` | Neurons, Activation Functions, Network Types | 21–63 |
| `Lecture_03_Linear_Regression.md` | Regression, Cost Function, Gradient Descent, Logistic | 64–123 |
| `Lecture_04_Simple_Neural_Network.md` | McCulloch-Pitts, Hebb, Perceptron, Adaline | 124–152 |
| `Lecture_05_Backpropagation.md` | Backprop, Chain Rule, Momentum, Optimizers, Dropout | 153–186 |
| `Lecture_06_CNN.md` | Convolution, Pooling, Feature Maps, CNN Pipeline | 187–235 |
| `Lecture_07_CNN_Architectures.md` | LeNet → AlexNet → VGG → GoogLeNet → ResNet → DenseNet | 235–260 |
| `Lecture_08_Object_Detection.md` | R-CNN, Fast/Faster R-CNN, YOLO, Mask R-CNN, Segmentation | 261–295 |
| `Lecture_09_RNN.md` | Sequential Data, RNN Architectures, BPTT, Vanishing Gradient | 296–312 |
| `Lecture_10_LSTM_GRU.md` | LSTM Gates, GRU, Bidirectional RNNs, Applications | 313–338 |
| `Lecture_11_Transformer.md` | Attention, Self-Attention, Positional Encoding, BERT, GPT | 339–379 |
| `Lecture_12_Deep_Generative_Modeling.md` | Autoencoders, VAE, GANs | 380–415 |

---

---

# المحاضرة الأولى — Introduction to Deep Learning

---

## Traditional Programming vs Machine Learning

| النموذج | الطريقة |
|---------|---------|
| **Traditional Programming** | انت بتدي الكمبيوتر الـ **Data** والـ **Rules**، وهو بيديك الـ **Output** |
| **Machine Learning** | انت بتدي الكمبيوتر الـ **Data** والـ **Output** المتوقع، وهو بيتعلم الـ **Rules** لوحده |

> يعني في البرمجة التقليدية انت بتكتب القواعد. في الـ ML الكمبيوتر بيكتشف القواعد لوحده من الـ data.

---

## أنواع الـ Machine Learning

| النوع | الطريقة | مثال |
|-------|---------|------|
| **Supervised** | training data فيها inputs + correct outputs | Classification, Regression |
| **Unsupervised** | data بدون labels — بيكتشف أنماط مخفية | Clustering |
| **Reinforcement** | بيتعلم عن طريق المكافآت والعقوبات | ألعاب AI |

---

## الهرم — AI → ML → DL

- **AI**: المجال الواسع — أي محاولة لخلي الماكينة ذكية
- **ML**: جزء من AI — التعلم من البيانات بدون برمجة صريحة
- **DL**: جزء من ML — شبكات عصبية عميقة بطبقات كتير

---

## Shallow vs Deep Learning

| الخاصية | Shallow | Deep |
|---------|---------|------|
| **Feature Extraction** | يدوي | تلقائي |
| **الأداء** | كويس على بيانات بسيطة | ممتاز على بيانات معقدة |

---

---

# المحاضرة التانية — Neural Network Basic

---

## الـ Artificial Neuron

```
x1 ──w1──┐
x2 ──w2──┤→ [ Σ(wi*xi) + b ] → [ f(a) ] → Output (y)
xn ──wn──┘
```

الـ neuron بياخد inputs × weights، يجمعهم + bias، ويطبّق activation function.

---

## Activation Functions

| الـ Function | الـ Range | المميزات | العيوب |
|-------------|-----------|----------|--------|
| **Sigmoid** | (0, 1) | كويس للـ probabilities | vanishing gradient |
| **Tanh** | (-1, 1) | zero-centered | vanishing gradient |
| **ReLU** | [0, +∞) | سريع، لا vanishing | dying ReLU |
| **Leaky ReLU** | (-∞, +∞) | بيحل dying ReLU | — |
| **Softmax** | (0, 1) مجموعهم 1 | multi-class classification | — |

---

## أنواع الشبكات

| النوع | الخاصية | الاستخدام |
|-------|---------|-----------|
| **Feedforward** | بيانات تمشي في اتجاه واحد | مشاكل عامة |
| **CNN** | فلاتر + pooling | صور وفيديو |
| **RNN** | feedback loop + ذاكرة | بيانات تسلسلية |
| **LSTM** | RNN + ذاكرة أطول | long-term dependencies |
| **GAN** | Generator vs Discriminator | توليد بيانات |
| **Transformer** | Self-Attention | NLP + رؤية حاسوبية |

---

---

# المحاضرة التالتة — Linear Regression

---

## الصيغ الأساسية

```
Linear Regression:     h(x) = θ₀ + θ₁ * x
Cost Function (MSE):   J(θ) = (1/2m) * Σ(h(xⁱ) - yⁱ)²
Gradient Descent:      θⱼ := θⱼ - α * ∂J/∂θⱼ
Logistic Regression:   h(x) = 1 / (1 + e^(-θᵀx))
```

## Feature Scaling

```
x_normalized = (x - x_min) / (x_max - x_min)
```

## Learning Rate

| α صغير أوي | التحديثات بطيئة جداً |
|------------|---------------------|
| **α كبير أوي** | **بيتخطى الـ minimum (overshoot)** |

---

---

# المحاضرة الرابعة — Simple Neural Network

---

## مقارنة الموديلات الأساسية

| الخاصية | McCulloch-Pitts | Hebb | Perceptron | Adaline |
|---------|----------------|------|------------|---------|
| **السنة** | 1943 | 1949 | 1957 | 1960 |
| **التعلم** | ❌ | ✅ | ✅ | ✅ |
| **القاعدة** | threshold يدوي | ∆w = x*y | ∆w = α(t-y)*x | ∆w = α(t-net)*x |
| **الـ Output** | binary | binary | binary | continuous |
| **الأساس** | logic gates | correlation | error correction | gradient descent |

---

## Classification

- **Binary**: output واحد → class 0 أو 1
- **Multi-class (K classes)**: K outputs + **one-hot encoding** (مثلاً [0,1,0,0] = class 2)

---

---

# المحاضرة الخامسة — Backpropagation Algorithm

---

## Backpropagation

| الخطوة | الاتجاه | الوظيفة |
|--------|---------|---------|
| **Forward** | Input → Output | حساب الـ output والـ loss |
| **Backward** | Output → Input | حساب الـ gradients وتعديل الـ weights |

---

## الـ Optimizers

| الـ Optimizer | الفكرة |
|-------------|--------|
| **SGD** | basic gradient descent |
| **SGD + Momentum** | بيضيف velocity — أسرع وأسلس |
| **Adagrad** | adaptive LR لكل parameter — مشكلة: LR بيقل كتير |
| **RMSprop** | moving average — مستقر + كويس لـ non-convex |
| **Adam** | Momentum + RMSprop — **الأفضل عموماً** |

---

## Dropout

- **في التدريب**: بيشيل نيورونات عشوائياً (مثلاً 50%) كل mini-batch
- **في الـ Testing**: مفيش dropout — الـ weights بتتضرب في (1-p)
- **الفايدة**: يمنع overfitting ويحسّن generalization

---

---

# المحاضرة السادسة — CNN

---

## بنية الـ CNN

```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

## حساب حجم الـ Output

```
Output Size = (N - F + 2P) / S + 1
```

## Convolution vs Pooling

| الطبقة | الوظيفة | الـ Parameters |
|--------|---------|---------------|
| **Convolution** | استخراج features باستخدام فلاتر | **بتتعلم** (weights) |
| **Pooling** | تقليل الحجم | **مفيش** parameters |

## Feature Hierarchy

```
Layer 1: Edges → Layer 2: Textures → Layer 3: Parts → Layer 4: Objects
```

---

---

# المحاضرة السابعة — CNN Architectures

---

## تطور المعماريات

| المعمارية | السنة | الطبقات | Parameters | Error | الابتكار |
|-----------|-------|---------|------------|-------|----------|
| **LeNet-5** | 1998 | 7 | 60K | — | أول CNN ناجح |
| **AlexNet** | 2012 | 8 | 60M | 16.4% | ReLU, Dropout, GPU |
| **ZFNet** | 2013 | 8 | — | 11.7% | فلاتر أصغر |
| **VGG-16** | 2014 | 16 | 138M | 7.3% | 3×3 فلاتر متكررة |
| **GoogLeNet** | 2014 | 22 | 5M | 6.7% | Inception Module |
| **ResNet** | 2015 | 152 | — | 3.57% | Skip Connections |
| **DenseNet** | 2017 | — | — | — | Dense Blocks |

### الأفكار الأساسية:
- **VGG**: 3×3 filters متكررة (أقل params من 5×5)
- **GoogLeNet**: Inception Module (1×1 + 3×3 + 5×5 + Pool في نفس الطبقة)
- **ResNet**: Skip Connections — `F(x) + x` (حل vanishing gradient)
- **DenseNet**: كل طبقة متصلة بكل اللي قبلها

---

---

# المحاضرة التامنة — Object Detection

---

## مهام الـ Computer Vision

| المهمة | الوصف |
|--------|-------|
| **Classification** | صورة → label |
| **Object Detection** | صورة → labels + bounding boxes |
| **Semantic Segmentation** | كل بكسل ← label (بدون تمييز instances) |
| **Instance Segmentation** | كل بكسل ← label + instance ID |

---

## تطور الـ Object Detection

| الموديل | الطريقة | السرعة |
|---------|---------|--------|
| **R-CNN** | Selective Search + CNN لكل region + SVM | بطيء جداً |
| **Fast R-CNN** | CNN واحد + ROI Pooling + Softmax | أسرع |
| **Faster R-CNN** | RPN (بدل Selective Search) | **الأسرع** |
| **YOLO** | Grid-based — مرة واحدة | **Real-time** |

---

## Segmentation

- **Semantic**: FCN — Down sampling → Up sampling + Skip Connections
- **Instance**: Mask R-CNN — Faster R-CNN + mask branch + ROI Align

---

---

# المحاضرة التاسعة — RNN

---

## بنية الـ RNN

```
h(t) = f(W_hh * h(t-1) + W_xh * x(t) + b)    ← hidden state
y(t) = W_hy * h(t) + b_y                        ← output
```

## أنواع المعماريات

| النوع | مثال |
|-------|------|
| **One-to-One** | Image Classification |
| **One-to-Many** | Image Captioning |
| **Many-to-One** | Sentiment Analysis |
| **Many-to-Many** | Machine Translation |

## Vanishing vs Exploding Gradient

| المشكلة | السبب | الحل |
|---------|-------|------|
| **Vanishing** | λ(W) < 1 | LSTM / GRU |
| **Exploding** | λ(W) > 1 | Gradient Clipping |

---

---

# المحاضرة العاشرة — LSTM & GRU

---

## LSTM Gates

| الـ Gate | الوظيفة |
|---------|---------|
| **Forget** | ايه ننسى من الذاكرة القديمة |
| **Input** | ايه نضيف من المعلومات الجديدة |
| **Output** | ايه نطلع كـ hidden state |

```
C(t) = f(t) × C(t-1) + i(t) × C̃(t)    ← Cell State update
h(t) = o(t) × tanh(C(t))                ← Hidden State
```

## RNN vs LSTM vs GRU

| الخاصية | RNN | LSTM | GRU |
|---------|-----|------|-----|
| **الذاكرة** | قصيرة | طويلة | طويلة |
| **الـ Gates** | 0 | 3 | 2 |
| **Cell State** | ❌ | ✅ | ❌ |
| **السرعة** | الأسرع | أبطأ | متوسط |

---

---

# المحاضرة الحادية عشر — Transformer

---

## الـ Attention

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

| المكون | الوظيفة |
|--------|---------|
| **Query** | الكلمة اللي بندور عنها |
| **Key** | الكلمات اللي بنقارن بيها |
| **Value** | المعلومة اللي عايزين ناخدها |

## المكونات الأساسية

| المكون | الوظيفة |
|--------|---------|
| **Positional Encoding** | يعرّف الموديل ترتيب الكلمات (Sine/Cosine) |
| **Multi-Head Attention** | كذا attention في نفس الوقت |
| **FFN** | طبقتين FC + ReLU بعد الـ attention |
| **Masked Attention** | يمنع الموديل يشوف الكلمات المستقبلية |
| **Add & Norm** | Residual Connection + Layer Normalization |

## أنواع الموديلات

| النوع | مثال | الاستخدام |
|-------|------|-----------|
| **Encoder-Decoder** | T5 | Translation, Summarization |
| **Encoder-Only** | **BERT** | Classification, QA |
| **Decoder-Only** | **GPT** | Text Generation |

---

---

# المحاضرة الثانية عشر — Deep Generative Modeling

---

## مقارنة الموديلات التوليدية

| الخاصية | Autoencoder | VAE | GAN |
|---------|-------------|-----|-----|
| **الهدف** | إعادة بناء | توليد | توليد واقعي |
| **الـ Latent** | نقطة | توزيع (μ, σ²) | random noise |
| **الـ Loss** | Reconstruction | Recon + KL | Adversarial |
| **جودة التوليد** | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **سهولة التدريب** | ⭐⭐⭐ | ⭐⭐ | ⭐ |

## GAN — Min-Max Game

```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

- **Generator**: بيحاول يخدع الـ Discriminator
- **Discriminator**: بيحاول يكشف المزيف

---

---

## أهم المعادلات والقوانين — Quick Reference

```
-- Linear Regression --
h(x) = θ₀ + θ₁x
J(θ) = (1/2m) * Σ(h(xⁱ) - yⁱ)²

-- Gradient Descent --
θⱼ := θⱼ - α * ∂J/∂θⱼ

-- Sigmoid --
σ(x) = 1 / (1 + e^(-x))

-- ReLU --
f(x) = max(0, x)

-- Softmax --
softmax(xi) = e^xi / Σ(e^xj)

-- CNN Output Size --
Output = (N - F + 2P) / S + 1

-- Perceptron Learning --
∆w = α * (t - y) * x

-- Adaline (Delta Rule) --
∆w = α * (t - net) * x

-- Hebb Rule --
∆w = x * y

-- Backpropagation (Chain Rule) --
dL/dw = dL/da * da/dz * dz/dw

-- Adam Optimizer --
m(t) = β₁*m(t-1) + (1-β₁)*∂L/∂w        (1st moment)
v(t) = β₂*v(t-1) + (1-β₂)*(∂L/∂w)²     (2nd moment)
w(t) = w(t-1) - α * m̂(t)/√(v̂(t)+ε)

-- LSTM --
f(t) = σ(W_f·[h(t-1),x(t)] + b_f)       (Forget gate)
i(t) = σ(W_i·[h(t-1),x(t)] + b_i)       (Input gate)
C(t) = f(t)*C(t-1) + i(t)*C̃(t)          (Cell state)
o(t) = σ(W_o·[h(t-1),x(t)] + b_o)       (Output gate)
h(t) = o(t) * tanh(C(t))                  (Hidden state)

-- Attention --
Attention(Q,K,V) = softmax(Q·Kᵀ/√d) · V

-- VAE Loss --
L = ||x - x̂||² + KL(q(z|x) || p(z))

-- GAN Objective --
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

---

> **Neural Network Course Revision — FCIS, Mansoura University**
