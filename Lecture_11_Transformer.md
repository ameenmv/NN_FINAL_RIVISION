# المحاضرة الحادية عشر — Transformer

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 339–379

---

## ما هو الـ Transformer؟

الـ **Transformer** هو نوع من الشبكات العصبية مصمم أساساً للمهام اللي فيها **sequence-to-sequence** (زي الترجمة: من إنجليزي لعربي).

### بيتكون من جزئين رئيسيين:

| الجزء | الوظيفة |
|-------|---------|
| **Encoder** | بيقرأ الجملة الأصلية (مثلاً الجملة بالإنجليزي) |
| **Decoder** | بيولّد الجملة اللي المفروض تخرج (مثلاً الترجمة بالعربي) |

```
Input: "The cat sat on the mat"
         ↓
   ┌──────────┐
   │ ENCODER  │ → Encoded Representation
   └────┬─────┘
        ↓
   ┌──────────┐
   │ DECODER  │ → "القطة جلست على الحصيرة"
   └──────────┘
```

### مكونات الـ Transformer:

```
┌─────────────────────────────────────────────┐
│                 ENCODER                      │
│  Input Embedding + Positional Encoding      │
│         ↓                                    │
│  Multi-Head Self-Attention                  │
│         ↓                                    │
│  Add & Norm                                 │
│         ↓                                    │
│  Feed Forward Network                       │
│         ↓                                    │
│  Add & Norm                                 │
│  (× N layers)                               │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│                 DECODER                      │
│  Output Embedding + Positional Encoding     │
│         ↓                                    │
│  Masked Multi-Head Self-Attention           │
│         ↓                                    │
│  Add & Norm                                 │
│         ↓                                    │
│  Multi-Head Cross-Attention (from Encoder)  │
│         ↓                                    │
│  Add & Norm                                 │
│         ↓                                    │
│  Feed Forward Network                       │
│         ↓                                    │
│  Add & Norm                                 │
│  (× N layers)                               │
│         ↓                                    │
│  Linear → Softmax → Output Probabilities    │
└─────────────────────────────────────────────┘
```

---

## الـ Attention Mechanism

الـ **Attention** هو آلية الموديل بيستخدمها عشان يعرف **ايه الجزء المهم** من الجملة يركّز عليه لما بيولّد كل كلمة.

> تخيّل عندك جدول فيه Keys و Values: الكلمة اللي الموديل بيولّدها بتمثل Query، والجملة الأصلية كلها بتمثل Keys و Values.

### الفكرة:

```
Query (الكلمة اللي بنولّدها)
  ↓
بيقارنها مع كل Key (كلمات الجملة)
  ↓
بيحسب درجة تشابه (weight من 0 لـ 1)
  ↓
بيضرب كل Value في الوزن بتاعها ويجمعهم
  ↓
= ده اللي بيستخدمه عشان يولّد الكلمة الجاية
```

---

## Queries, Keys, Values (Q, K, V)

| المكون | المعنى | الوظيفة |
|--------|--------|---------|
| **Query (Q)** | الكلمة اللي بندور عنها | بتمثل embeddings بدون context |
| **Key (K)** | الكلمات اللي بنقارن بيها | بتمثل الـ surrounding embeddings (context) |
| **Value (V)** | المعلومة اللي عايزين ناخدها | embeddings مع context (الـ output) |

---

## Scaled Dot-Product Attention

الـ attention score بيتحسب باستخدام **Dot Product** بين الـ Q والـ K:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

| الخطوة | الوصف |
|--------|-------|
| `Q · Kᵀ` | حساب التشابه بين كل Query وكل Key |
| `/ √d` | **تقسيم على الجذر التربيعي لـ d** (أبعاد الـ embeddings) |
| `softmax()` | تحويل الأوزان لاحتمالات (مجموعها 1) |
| `× V` | ضرب كل Value في وزنها وجمعهم |

### ليه القسمة على √d؟

> لما الأبعاد تكون كبيرة، الـ Dot Product بيبقى كبير جداً وده بيبوّظ الـ Softmax ويخلي التدريب صعب. القسمة بتعمل scaling عشان القيم تكون معقولة.

---

## Multi-Head Self-Attention

**مش بنكتفي بـ Attention واحدة** — بنعمل كذا Attention **في نفس الوقت** (multi-head).

```
         Input
    ┌──────┼──────┐
    ↓      ↓      ↓
  Head₁  Head₂  Head₃     ← كل "رأس" عندها Q, K, V مختلفة
    ↓      ↓      ↓
    └──────┼──────┘
      Concatenate
           ↓
     Linear Transform
           ↓
        Output
```

| الخاصية | التفاصيل |
|---------|----------|
| **كل head** | عندها **weight matrices** مختلفة (W_Q, W_K, W_V) |
| **الميزة** | كل head بتبص على **جوانب مختلفة** من الجملة |
| **النتيجة** | بنجمع نتايج كل الرؤوس (**concatenation**) وبندخلهم على Linear layer |
| **التوازي** | الـ outputs بتتحسب **بالتوازي** |

---

## Positional Encoding

الـ Transformer **مش بيبقى عارف ترتيب الكلمات** لوحده — عكس الـ RNN اللي بيمشي خطوة خطوة.

### الحل:

بنضيف **Positional Encoding** — أرقام بتتحسب من **موقع الكلمة** في الجملة باستخدام دوال **Sine** و **Cosine**.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

| الرمز | المعنى |
|-------|--------|
| **pos** | موقع الـ token في الجملة |
| **i** | بُعد الـ encoding |
| **d** | حجم الـ embedding |

> علشان كده بنضيف الـ Positional Encoding على الـ input embeddings عشان نعرّف الموديل ترتيب الكلمات.

---

## Feed-Forward Network (FFN)

بعد الـ Attention، بنعدّي كل كلمة على **شبكة عصبية صغيرة** (FFN) — بتتكون من **طبقتين Fully Connected** وفي النص بينهم **ReLU**.

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

| الخاصية | التفاصيل |
|---------|----------|
| **Position-wise** | بتتطبّق على **كل كلمة لوحدها** |
| **الوظيفة** | بتضيف **non-linearity** بعد الـ attention |
| **الطبقات** | الأولى حجم متغير، التانية بنفس حجم الـ embedding |

> ودي بتحسّن التمثيل النهائي لكل كلمة بعد ما اتجمعت معلوماتها من باقي الجملة.

---

## Masked Self-Attention (للـ Decoder)

لو بنعمل توليد جملة (زي في الترجمة)، **ماينفعش** الموديل يشوف الكلمات اللي **بعد** الكلمة الحالية.

```
"The cat sat on the ___"
  ✓    ✓   ✓   ✓   ✓   ✗ ✗ ✗   ← الموديل يشوف بس لحد الكلمة الحالية
```

| الخاصية | التفاصيل |
|---------|----------|
| **الطريقة** | بنحط **mask** — الـ attention weights للكلمات المستقبلية = 0 (أو رقم سالب كبير قبل softmax) |
| **الهدف** | الموديل يبني الجملة **واحدة واحدة** — من الشمال لليمين |

---

## Residual Connections + Layer Normalization

### Residual Connections:

بدل ما نعدّي البيانات على Layer بس، **بنضيف الناتج على نفس البيانات الأصلية** — عشان الموديل يتعلم **الفرق** بس.

```
output = LayerNorm(x + SubLayer(x))
```

### Layer Normalization:

بتنظّم القيم عشان التدريب يبقى **ثابت ومستقر**. بتشتغل على **كل كلمة لوحدها** مش على الـ batch كله.

> في رسومات الـ Transformer بتلاقيها مكتوبة **"Add & Norm"** — وده يعني Residual Connection + Layer Normalization.

---

## أنواع الـ Transformer Models

### 1. Encoder-Decoder (Full Transformer):

| الخاصية | التفاصيل |
|---------|----------|
| **الاستخدام** | Machine Translation, Summarization |
| **الـ Encoder** | بيقرأ الجملة الأصلية (**bidirectional**) |
| **الـ Decoder** | بيستخدم output الـ encoder + الكلمات اللي ترجمها لحد دلوقتي (**unidirectional**) |
| **Cross-Attention** | الـ Decoder بيعمل attention على output الـ Encoder (Keys, Values من Encoder / Queries من Decoder) |

### 2. Encoder-Only — BERT:

| الخاصية | التفاصيل |
|---------|----------|
| **الاستخدام** | Classification, Extractive QA, Image Classification |
| **الطريقة** | بيفهم الكلام بس — **مش بيولّد جمل** |
| **التدريب** | **Masked Language Model** — بيشيل كلمات (Mask) والموديل يحاول يكمّلها |
| **مهمة تانية** | **Next Sentence Prediction** — بيشوف جملتين ويسأل: التانية بتيجي بعد الأولى ولا لأ؟ |

> موديلات الـ Encoder بس زي **BERT** مش بتولّد جمل — هي بتفهم الكلام بس. بتستخدم في تصنيف الجمل والأسئلة وغيرها.

### 3. Decoder-Only — GPT:

| الخاصية | التفاصيل |
|---------|----------|
| **الاستخدام** | **Text Generation** — توليد النصوص |
| **الطريقة** | **Auto-regressive** — بيشوف الكلمات اللي قبل الكلمة اللي عايز يتوقعها |
| **التدريب** | بيتوقع الـ **next token** في السلسلة |
| **أمثلة** | GPT-2 (1.5B params), GPT-3 (175B params), ChatGPT |

> موديلات الـ Decoder بس زي **GPT** بتشتغل على توليد النص. بتبص على الكلمات اللي قبل الكلمة وبتحاول تتوقع الكلمة الجاية. زي ما بيحصل في ChatGPT لما بيكتبلك رد.

---

## Transformers for Image Classification — Vision Transformer (ViT)

مش بس نصوص — الـ Transformers ممكن تشتغل على **صور** كمان!

### الطريقة:

```
Image (224×224)
      ↓
Split into 16×16 patches    ← الصورة بتتقطع لأجزاء صغيرة (patches)
      ↓
Flatten each patch           ← كل patch بيتحول لـ vector
      ↓
Linear Projection            ← بيتحول لـ embedding
      ↓
Add [class] token            ← patch مخصوص بيتعلم يمثل الصورة كلها
      ↓
Add Positional Encoding      ← عشان الموديل يعرف مكان كل patch
      ↓
Transformer Encoder          ← بنستخدم الـ Transformer عليهم كأنهم جملة
      ↓
[class] token → Classification Head → Output
```

> يعني الصورة بتتقطع لأجزاء صغيرة (patches)، وكل patch بيتحول لرقم (vector). بعد كده بنستخدم الـ Transformer عليهم كأنهم جملة. وفيه patch مخصوص بيتعلم يمثّل الصورة كلها وبنستخدمه في الآخر عشان نصنّف الصورة.

---

## مقارنة — RNN vs Transformer

| الخاصية | RNN/LSTM | Transformer |
|---------|----------|-------------|
| **المعالجة** | تسلسلية (واحدة واحدة) | **بالتوازي** |
| **الذاكرة** | hidden state | **Self-Attention** (بتشوف كل الكلمات) |
| **الـ Context** | قصير (حتى LSTM) | **طويل جداً** |
| **السرعة** | بطيء | **سريع** (parallelizable) |
| **الـ Long Dependencies** | صعبة | **سهلة** (attention) |

---

> **نهاية المحاضرة الحادية عشر**
