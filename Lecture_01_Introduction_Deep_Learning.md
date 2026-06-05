# المحاضرة الأولى — Introduction to Deep Learning

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 1–20

---

## 1. What is Machine Learning?

> **"A field of study that gives computers the ability to learn without being explicitly programmed"** — Arthur Samuel

## Traditional Programming vs Machine Learning

في البداية لازم نفهم الفرق الجوهري بين الطريقة التقليدية والـ Machine Learning:

| النموذج | الطريقة | المثال |
|---------|---------|--------|
| **Traditional Programming** | انت بتدي الكمبيوتر الـ **Data** والـ **Rules** (البرنامج)، وهو بيديك الـ **Output** | `if temperature > 30: print("Hot")` |
| **Machine Learning** | انت بتدي الكمبيوتر الـ **Data** والـ **Output** المتوقع، وهو بيتعلم الـ **Rules** لوحده | بتديله صور قطط وكلاب، وهو بيتعلم يفرق بينهم |

```
Traditional Programming:          Machine Learning:
┌──────────┐                      ┌──────────┐
│   Data   │──┐                   │   Data   │──┐
└──────────┘  ├──→ [Computer] → Output    ├──→ [Computer] → Program (Rules)
┌──────────┐  │                   ┌──────────┐  │
│  Program │──┘                   │  Output  │──┘
│  (Rules) │                      │          │
└──────────┘                      └──────────┘
```

> يعني ببساطة: في البرمجة التقليدية انت بتكتب القواعد. في الـ ML الكمبيوتر بيكتشف القواعد لوحده من الـ data.

### مثال: Activity Recognition

تخيل عايز تعرف الشخص بيعمل ايه (ماشي، بيجري، بيسوق دراجة):

```
الطريقة التقليدية:
if(speed < 4)    → WALKING
if(speed < 12)   → RUNNING
else              → BIKING
... 🤔 ايه بقا لو الحركة أعقد من كده؟

الطريقة ML:
بتدي الكمبيوتر sensor data + labels → وهو بيتعلم يصنّف لوحده ✅
```

---

## 2. أنواع الـ Machine Learning

### Supervised Learning (تعلم بإشراف):
- الـ training data فيها **inputs + correct outputs** (labels)
- المهام الأساسية:
  - **Classification** — تصنيف (مثلاً: صورة قطة ولا كلب؟)
  - **Regression** — توقع قيمة مستمرة (مثلاً: سعر بيت)

### Unsupervised Learning (تعلم بدون إشراف):
- الـ data **مفيهاش labels** — مفيش إجابة صح
- الهدف: **اكتشاف أنماط مخفية** أو تجميع البيانات المتشابهة
- أنواعه:
  - **Clustering** — تقسيم العملاء لمجموعات
  - **Dimensionality Reduction** — تقليل الأبعاد مع الحفاظ على المعلومات المهمة

### Reinforcement Learning (تعلم بالتعزيز):
- الموديل بيتعلم عن طريق **المكافآت والعقوبات** بناءً على أفعاله
- بيتعلم يعمل **sequence of actions** تعظّم الـ reward
- مثال: تعليم AI يلعب لعبة، Robot Navigation، Game AI

---

## 3. الهرم — AI vs ML vs DL

```
┌──────────────────────────────────────┐
│         AI (Artificial Intelligence) │  ← المجال الواسع: خلي الماكينة ذكية
│  ┌──────────────────────────────┐    │
│  │    ML (Machine Learning)     │    │  ← جزء من AI: التعلم من البيانات
│  │  ┌──────────────────────┐    │    │
│  │  │  DL (Deep Learning)  │    │    │  ← جزء من ML: شبكات عصبية عميقة
│  │  └──────────────────────┘    │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

| المستوى | التعريف |
|---------|---------|
| **AI** | المجال الواسع — أي محاولة لخلي الماكينة تتصرف بذكاء |
| **ML** | جزء من AI — الماكينة بتتعلم من البيانات بدون ما تتبرمج صريح |
| **DL** | جزء من ML — بيستخدم شبكات عصبية عميقة بطبقات كتير، بتتعلم من كميات ضخمة من البيانات |

---

## 4. ML vs DL — الفرق التفصيلي

| الخاصية | Machine Learning | Deep Learning |
|---------|-----------------|---------------|
| **Feature Extraction** | **يدوي** — انت بتختار الـ features بإيدك (SIFT, HOG) | **تلقائي** — الموديل بيتعلم الـ features لوحده |
| **الـ Classifier** | Generic classifiers زي **SVM, KNN** | شبكات عصبية عميقة بطبقات كتير |
| **الأداء** | كويس على بيانات بسيطة | **ممتاز** على بيانات معقدة (صور، نصوص، صوت) |
| **الاحتياج للبيانات** | بيانات أقل | **بيانات كتيييير** |

> يعني ببساطة: في الـ ML كنا بنعمل feature extraction بإيدينا (SIFT, HOG, etc.) وبعدين نديها لـ classifier زي SVM. لكن في الـ DL الموديل بيعمل feature extraction + classification كله لوحده.

---

## 5. Deep Learning = Hierarchical Compositionality

الـ DL بيتعلم الـ features في **مستويات** (hierarchy):

```
Input Image → [Low-Level Features] → [Mid-Level Features] → [High-Level Features] → Class
                    ↓                       ↓                        ↓
               Edges, dark spots      Eyes, ears, nose         Facial structure
```

> يعني ببساطة: الطبقات الأولى بتتعلم حواف وخطوط بسيطة، الطبقات الوسطى بتتعلم أجزاء أكبر (عين، أنف)، والطبقات العميقة بتتعلم أشكال كاملة (وش كامل). ده مبني على ورقة Zeiler & Fergus 2014.

---

## 6. تاريخ الـ Deep Learning — Brief History

### المرحلة الأولى (1943–2006):

| السنة | الحدث |
|-------|-------|
| **1943** | **McCulloch & Pitts** — أول نموذج رياضي لـ artificial neuron |
| **1957** | **Rosenblatt** — الـ **Perceptron** — أول نموذج تعلم |
| **1960** | **Widrow & Hoff** — **ADALINE** و **MADALINE** — أول تطبيقات عملية |
| **1969** | **Minsky & Papert** — كتاب "Perceptrons" — بيوضح حدود الـ single-layer (مشكلة **XOR**) → **AI Winter** |
| **1986** | **Rumelhart, Hinton & Williams** — أعادوا اكتشاف **Backpropagation** → تدريب شبكات متعددة الطبقات |
| **1995** | **Vapnik & Cortes** — **SVMs** بتظهر وبتنافس الشبكات العصبية |
| **2006** | **Hinton** — **Deep Belief Networks** + Pretraining → عصر الـ DL بيبدأ |

### المرحلة الثانية (2012–2023):

| السنة | الحدث |
|-------|-------|
| **2012** | **AlexNet** (Krizhevsky, Sutskever, Hinton) بيكسب **ImageNet** — الـ CNNs بتنتشر |
| **2014** | **GANs** — Ian Goodfellow بيقدم الـ Generative Adversarial Networks |
| **2015** | **ResNet** — شبكات **152 طبقة** باستخدام Skip Connections |
| **2016** | **AlphaGo** (Google DeepMind) بيهزم بطل العالم **Lee Sedol** في Go — Deep RL |
| **2017** | **Transformers** (Vaswani et al., Google Brain) — Self-Attention بدل الـ RNNs |
| **2018** | **BERT** (Google) — **Bidirectional** pre-training + fine-tuning → Transfer Learning في NLP |
| **2020** | **GPT-2, GPT-3** (OpenAI) — 175 billion parameters — text generation, translation, code |
| **2022** | **ChatGPT** — Fine-tuned GPT-3.5 باستخدام **RLHF** (Reinforcement Learning from Human Feedback) |
| **2023** | **GPT-4** — **Multimodal** (text + images) — فاهم صور وبيولّد نصوص |

---

## 7. Why Now? — ليه DL انتشر دلوقتي؟

الشبكات العصبية موجودة من عقود، فليه بقت مسيطرة دلوقتي؟

| السبب | التفاصيل |
|-------|----------|
| **1. Big Data** | بيانات أكبر بكتير — أسهل في التجميع والتخزين (ImageNet, Wikipedia) |
| **2. Hardware (GPUs)** | كروت الشاشة بتعمل **Massively Parallel** computations — أسرع بـ 50x من CPU |
| **3. Software** | أدوات وتقنيات جديدة: **TensorFlow, PyTorch** + تقنيات محسّنة (Batch Norm, Dropout, Adam) |

> يعني ببساطة: الفكرة كانت موجودة من زمان، بس محتاجة **بيانات كتير + أجهزة قوية + أدوات كويسة** عشان تشتغل. الـ 3 دول اتوفروا بعد 2010.

---

## 8. ImageNet Challenge (ILSVRC)

- قاعدة بيانات فيها **~14 مليون صورة** مصنّفة في **20,000 class**
- التحدي: **1.2 مليون training image** في **1000 class**
- المطلوب: classification — تقول الصورة فيها ايه

```
Error Rate عبر السنين (Top-5 Error %):

2010-2011  ████████████████████████████  28.2% - 25.8%  (Shallow methods)
2012       ████████████████             16.4%  ← AlexNet (8 layers) 💥
2013       ███████████                  11.7%  ← ZFNet (8 layers)
2014       ███████                       7.3%  ← VGG (19 layers)
2014       ██████                        6.7%  ← GoogLeNet (22 layers)
2015       ████                          3.57% ← ResNet (152 layers) 🏆
```

> يعني ببساطة: كل سنة الـ Deep Learning بيبقى أحسن. في 2015 الـ ResNet تفوّق على مستوى البشر في التعرف على الصور!

---

> **نهاية المحاضرة الأولى**
