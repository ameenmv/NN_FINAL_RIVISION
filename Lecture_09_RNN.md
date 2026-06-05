# المحاضرة التاسعة — Recurrent Neural Networks (RNN)

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 296–312

---

## ليه RNN؟

الشبكات العادية (CNN, Feedforward) بتتعامل مع كل input **مستقل** — مفيش ذاكرة. لكن في كتير من المشاكل الـ **output بيعتمد على المدخلات السابقة**.

> يعني لو عايز أترجم جملة، لازم الموديل يفتكر الكلمات اللي فاتت عشان يقدر يحدد الكلمة الجاية.

### أمثلة:
- **Speech Recognition**: "Hello" → "Hallo" (ترجمة)
- **Sentiment Analysis**: "I loved the movie" → Positive
- **Language Modeling**: توقع الكلمة الجاية في جملة

---

## بنية الـ RNN

الـ **RNN** عندها **feedback loop** — الـ hidden state من الخطوة السابقة بيدخل مع الـ input الجديد.

```
            ┌──────────────────────────────┐
            │                              │
            ↓                              │
x(t) ──→ ┌─────┐ ──→ y(t)                │
          │ RNN │                           │
h(t-1)──→│     │──→ h(t) ─────────────────┘
          └─────┘
```

### لو فردناها على الزمن (Unrolled):

```
x(1)      x(2)      x(3)          x(T)
  ↓         ↓         ↓              ↓
┌─────┐  ┌─────┐  ┌─────┐      ┌─────┐
│ RNN │→ │ RNN │→ │ RNN │→ ...→│ RNN │
└──┬──┘  └──┬──┘  └──┬──┘      └──┬──┘
   ↓         ↓         ↓              ↓
  y(1)      y(2)      y(3)          y(T)
```

### المعادلات:

```
h(t) = f(W_hh * h(t-1) + W_xh * x(t) + b_h)    ← الـ hidden state الجديد
y(t) = W_hy * h(t) + b_y                          ← الـ output
```

| الرمز | المعنى |
|-------|--------|
| **h(t)** | الـ hidden state في الوقت t — **الذاكرة** |
| **x(t)** | الـ input في الوقت t |
| **W_hh** | weights من hidden لـ hidden |
| **W_xh** | weights من input لـ hidden |
| **W_hy** | weights من hidden لـ output |

> بتاخد الحالة السابقة من الـ hidden layer وتدخلها مع المدخل الجديد. الـ hidden layers بتخزن المعلومات اللي فاتت.

### الـ Parameter Sharing:
الـ RNN بتستخدم **نفس الـ weights** (W, U, V) في **كل خطوة زمنية** — ده بيخليها تقدر تتعامل مع sequences بأطوال مختلفة. لو مفيش parameter sharing، كنا هنحتاج weights مختلفة لكل طول sequence — ومش هنقدر نعمم على أطوال ماشفناهاش.

---

## أنواع المهام التسلسلية

| النوع | الوصف | مثال |
|-------|-------|------|
| **Sequence Recognition** | النموذج بيتعرف على **نمط معين** بناءً على سلسلة من المدخلات | تحليل المشاعر — جملة → "Positive" |
| **Sequence Reproduction** | النموذج بيكمّل السلسلة بعد ما يشوف أول جزء منها | توقع سعر الأسهم |
| **Temporal Association** | بيطلع سلسلة من المخرجات بناءً على سلسلة من المدخلات | الترجمة من لغة للغة |

---

## معماريات الـ RNN

| النوع | الـ Input | الـ Output | مثال |
|-------|----------|-----------|------|
| **One-to-One** | واحد | واحد | Image Classification (عادي) |
| **One-to-Many** | واحد | سلسلة | Image Captioning — صورة → وصف |
| **Many-to-One** | سلسلة | واحد | Sentiment Analysis — جملة → إيجابي/سلبي |
| **Many-to-Many** | سلسلة | سلسلة | Machine Translation — إنجليزي → عربي |
| **Synchronized Many-to-Many** | سلسلة | سلسلة (متزامنة) | Video Classification — كل frame ← label |

```
One-to-One:    One-to-Many:    Many-to-One:    Many-to-Many:
  ┌─┐            ┌─┐─┬─┬─┐     ┌─┬─┬─┐─┐      ┌─┬─┬─┐ ┌─┬─┬─┐
  │ │→○          │ │→○→○→○     │→│→│→│ │→○    │→│→│→│→│→│→│→○
  └─┘            └─┘           └─┴─┴─┘─┘      └─┴─┴─┘ └─┴─┴─┘
                                                Encoder  Decoder
```

---

## تدريب الـ RNN — BPTT

الـ RNN بيتدرب باستخدام **Backpropagation Through Time (BPTT)** — نسخة من الـ backpropagation بس بتمشي **عبر الزمن** مش بس عبر الطبقات.

> يعني بنفرد الشبكة على كل الخطوات الزمنية وبنعمل backprop على كل الخطوات.

---

## مشكلة الـ Vanishing / Exploding Gradients

### المشكلة:

أثناء الـ backpropagation عبر الزمن، الـ gradients بتفضل تتضرب في الـ weight matrix **مرة بعد مرة**. لو الـ weights صغيرة → الـ gradient بيختفي. لو كبيرة → الـ gradient بينفجر.

| المشكلة | متى بتحصل | النتيجة |
|---------|-----------|---------|
| **Vanishing Gradient** | لما أكبر **eigenvalue (λ)** لمصفوفة الأوزان W **أقل من 1** | التدرجات صغيرة جداً → الموديل مش بيتعلم على التسلسلات الطويلة |
| **Exploding Gradient** | لما أكبر **eigenvalue (λ)** لمصفوفة الأوزان W **أكبر من 1** | التدرجات كبيرة جداً → NaN/overflow |

### التفسير الرياضي (من الكتاب):

```
dS_t/dS_k = ∏(i=k+1 to t) dS_i/dS_{i-1}

dS_t/dS_{t-1} = diag(1 - tanh(...)^2) × W^T

لو λ > 1 → الـ product بينفجر (explode)
لو λ < 1 → الـ product بيختفي (vanish)
```

> يعني الـ gradient بيتضرب في مصفوفتين: W ومشتقات الـ activation function. تكرار الضرب ده لـ (t-k) مرة بيخلي الـ gradient ينفجر أو يختفي.

### مثال على تأثير الـ Vanishing:

```
"I grew up in France... [100 كلمة]... I speak fluent ______"
                                                        ↑
                                          الـ RNN مش هيقدر يربط
                                          "France" بالإجابة "French"
                                          لأن الـ gradient اختفى
```

### الحل:

| الحل | الوصف |
|------|-------|
| **Gradient Clipping** | لما الـ gradient يوصل لرقم معين، بنرجعه لعدد صغير |
| **LSTM / GRU** | شبكات مصممة خصيصاً عشان تتعامل مع المشكلة دي (المحاضرة الجاية) |

> **الـ GRNNs (Gated RNNs)** هي نوع خاص من الـ RNN عندها القدرة تتعلم **long-term dependencies** عن طريق **gating mechanisms** — الـ gates بتتحكم بشكل انتقائي في تدفق المعلومات. زي مفاتيح — بتسمح للمعلومات تعدّي بس لو بتلبي شروط معينة.

---

> **نهاية المحاضرة التاسعة**
