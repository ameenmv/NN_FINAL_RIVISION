# المحاضرة التانية — Neural Network Basic

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 21–63

---

## تعريف الـ Neural Network

الـ **Artificial Neural Network (ANN)** هي نظام معالجة معلومات مستوحى من **الشبكات العصبية البيولوجية** في مخ الإنسان.

> يعني ببساطة: بنحاول نحاكي الطريقة اللي المخ البشري بيشتغل بيها عشان الكمبيوتر يقدر يتعلم ويتعرف على أنماط.

### الافتراضات الأساسية:

- المعلومات بتتعالج عن طريق وحدات بسيطة اسمها **Neurons** — الوحدة الأصغر المكوّنة للشبكة هي النيورون
- النيورونات متوصلة ببعض عن طريق **Links** بتنقل إشارات بينهم
- كل link عليه **Weight** بيضرب في الإشارة — الـ `W` بيكون على كل رابط
- النيورونات بتطبق **Activation Functions** عشان تحدد الـ output

---

## الـ Artificial Neuron — النيورون الصناعي

النيورون هو الـ **building block** الأساسي لأي شبكة عصبية. بياخد inputs، بيطبق عليهم weights و bias، وبيستخدم activation function عشان يطلع output.

```
  x1 ──w1──┐
  x2 ──w2──┤
  x3 ──w3──┼──→ [ Σ (wi*xi) + b ] ──→ [ f(a) ] ──→ Output (y)
  ...      │         Weighted Sum       Activation
  xn ──wn──┘              ↑                Function
                          b (bias)
```

### إزاي بنحسب الـ Output:

```
a = Σ(wi * xi) + b        ← الـ weighted sum + bias
y = f(a)                   ← الـ activation function بتطبّق على a
```

> يعني بنضرب الـ weight بتاع كل link في الـ x (القيمة اللي داخلة) ونجمعهم كلهم على الـ bias اللي هو b.

---

## أهمية الـ Bias Unit

الـ **Bias** وظيفته إنه **يحرّك الـ activation function** يمين أو شمال عشان تبقى مناسبة للـ data اللي عندنا.

| الخاصية | التفاصيل |
|---------|----------|
| **الوظيفة** | بيحرّك الـ function يمين أو شمال |
| **الفايدة** | بيساعد النيورون يعمل fit أفضل للـ data |
| **التأثير** | بيؤثر على الـ output النهائي لكنه **مبيأثرش مباشرة على المدخلات** |

---

## أهمية الـ Activation Functions

الـ Activation Functions بتدخل **non-linearity** في الشبكة — وده اللي بيخليها تقدر تحل **مشاكل معقدة** مش بس المشاكل الخطية (Linear).

> يعني لولا الـ activation functions، الشبكة كلها هتبقى عبارة عن linear function واحدة مهما كان عدد الطبقات.

---

## أنواع الـ Activation Functions

### 1. Linear Activation:
- مفيش squashing — الـ output = الـ input
- **مش مفيد** في الشبكات العميقة — أي تغيير بعمله على المدخلات مفيش فرق

### 2. Binary Step Function:
- بيشتغل بالـ **threshold** — لو القيمة أكبر من الـ threshold يطلع `1`، غير كده يطلع `0`
- الـ output بيكون **0 أو 1** (أو -1 و 1)

### 3. Sigmoid Function:

```
σ(x) = 1 / (1 + e^(-x))
```

| الخاصية | التفاصيل |
|---------|----------|
| **الـ Output** | بين **0 و 1** |
| **الاستخدام** | كويس للـ **probabilities** |
| **الـ Threshold** | لو القيمة ≥ 0.5 → output = 1, لو أقل من 0.5 → output = 0 |
| **المشاكل** | **Vanishing Gradient** + حسابياً مكلف |

> مهم جداً حفظ المعادلة دي.

**مشكلة الـ Vanishing Gradient:**
لما القيم تكون كبيرة أوي أو صغيرة أوي، التغيير في الـ prediction بيكون **شبه معدوم** — يعني الـ update في الـ weight قليل جداً فمفيش أي learn جديد.

### 4. Hyperbolic Tangent (Tanh):

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

| الخاصية | التفاصيل |
|---------|----------|
| **الـ Output** | بين **-1 و 1** |
| **الميزة** | **Zero-centered** — بيؤدي لـ convergence أسرع من Sigmoid |
| **العيب** | لسه بيعاني من **Vanishing Gradient** عند القيم العالية/المنخفضة |

### 5. ReLU (Rectified Linear Unit):

```
f(x) = max(0, x)
```

| الخاصية | التفاصيل |
|---------|----------|
| **الـ Output** | **0** لو x ≤ 0، و **x** لو x > 0 |
| **المميزات** | يشجع **sparse activations**، أسرع في الـ convergence، بيتجنب vanishing gradient |
| **المشكلة** | **Dying ReLU** — لو النيورون طلع 0، ممكن يتوقف عن التعلم نهائي |
| **الحلول** | **Leaky ReLU** أو **Parametric ReLU** |

> يعني أي قيمة أصغر من 0 بيخليها 0، غير كده بيسيبها زي ما هي.

### مقارنة شاملة:

| الـ Function | الـ Range | المميزات | العيوب |
|-------------|-----------|----------|--------|
| **Linear** | (-∞, +∞) | بسيط | مش مفيد في deep networks |
| **Binary Step** | {0, 1} | بسيط | مش differentiable |
| **Sigmoid** | (0, 1) | كويس للـ probabilities | vanishing gradient |
| **Tanh** | (-1, 1) | zero-centered | vanishing gradient |
| **ReLU** | [0, +∞) | سريع، مفيش vanishing | dying ReLU |
| **Leaky ReLU** | (-∞, +∞) | بيحل dying ReLU | — |

---

## خصائص الـ Neural Network — Neural Network Characterization

| الخاصية | التفاصيل |
|---------|----------|
| **Architecture** | الطريقة اللي النيورونات هيتوصلوا ببعض بيها (neuron connection pattern) |
| **Activation Function** | الـ function اللي بتحدد الـ output |
| **Training Algorithm** | الطريقة اللي الـ weights بتتعلم بيها |
| **Hyperparameters** | حاجات زي **learning rate**, **epochs**, **batch size** |

---

## قدرة النيورون الواحد — Capacity of a Single Neuron

- النيورون الواحد يقدر يعمل **binary classification** باستخدام Sigmoid (زي logistic regression)
- يقدر يحل المشاكل الـ **linearly separable** بس
- **مش بيقدر** يحل المشاكل الـ **non-linearly separable** (زي **XOR**) إلا لو حوّلنا الـ input

> يعني نيورون واحد شغّال على المشاكل الخطية بس.

---

## أنواع الشبكات العصبية

### 1. Multilayer Neural Network (Feedforward):
- شبكة بـ **input layer, hidden layers, output layer**
- بتبدأ من قدام لحد الـ output — **Fully Connected** (كل نيورون متوصل بكل النيورونات في الـ layer اللي بعده)
- بتحل مشاكل أعقد من الشبكات ذات الطبقة الواحدة
- التدريب بيبقى **أصعب وأكتر استهلاك للموارد**

### 2. Convolutional Neural Networks (CNNs):
- شبكات feedforward متخصصة بـ **minimal preprocessing**
- بتتعلم **spatial features** تلقائياً (حواف، textures)
- **Convolution Layer**: بتستخدم فلتر عشان تميز حواف الصورة والشخصية
  - Hyperparameters: **filter size (F)**, **stride (S)**, **padding (P)**
- **Pooling Layer**: تقليل حجم الصورة بدون فقد معلومات
  - **Max Pooling**, **Average Pooling**
- بتستخدم في **image/video recognition** والـ **NLP**

### 3. Recurrent Neural Networks (RNNs):
- عندها **feedback loops** وبتحافظ على **hidden state** (ذاكرة)
- بتتعامل مع البيانات المعتمدة على بعض (sequential data)
- مفيدة في: **language modeling**, **time-series**, إلخ
- مشكلتها: **short-term memory** — بتنسى الـ data اللي مخزنتها بسررررعة

### 4. Long Short-Term Memory (LSTM):
- نفس الـ RNN بس **زوّدوا حجم الذاكرة**
- مصممة تتعامل مع **long-term dependencies**
- بتستخدم **cell state** و **gates** للتحكم في المعلومات:
  - **Input Gate**: ايه اللي نحتفظ بيه من الـ input
  - **Forget Gate**: ايه اللي ننساه من الحالة السابقة
  - **Output Gate**: ايه اللي نطلعه كـ output

### 5. Generative Adversarial Networks (GANs):
- **Generator**: بيولّد data مزيفة
- **Discriminator**: بيحاول يفرق بين المزيف والحقيقي
- بتستخدم في **توليد صور وفيديوهات وصوت واقعي**

### 6. Transformer Networks:
- العمود الفقري للـ **NLP** الحديث
- بتستخدم **Self-Attention** عشان تتعامل مع الـ input sequences
- عندها القدرة على **تمييز الكلمات المهمة** في الجملة
- بتقدر تعالج **بالتوازي** (مش زي RNN)
- بنية **Encoder-Decoder**
- بتستخدم في موديلات زي **BERT, GPT**

---

## الـ Softmax Activation Function

```
softmax(xi) = e^xi / Σ(e^xj)    for j = 1 to K
```

- بتستخدم في مشاكل **multi-class classification**
- بتطلع **أكتر من output**، كل واحد بيمثل احتمال class معين
- مجموع الـ outputs = **1** وكلهم **موجبين**
- أكبر احتمال هو الـ predicted class

> يعني لو مديله صورة موزة، ممكن يقول: تفاح 0.1، موز 0.7، بطيخ 0.2 — أكبر احتمال هو الموز فده الناتج.

---

## قدرة الشبكات العصبية — Capacity of Neural Networks

| عدد الطبقات | القدرة |
|-------------|--------|
| **Hidden layer واحدة** | بتقدر تتعامل مع مشاكل **بسيطة نسبياً** (زي المشاكل الخطية) |
| **شبكات عميقة (Deep)** | بتقدر تتعامل مع **أنماط أعقد بكتير** |

---

## الشبكات العصبية البيولوجية — Biological Neural Networks

### Visual Cortex (القشرة البصرية):

| المنطقة | الوظيفة |
|---------|---------|
| **V1 neurons** | بتكتشف الأشكال البسيطة زي **الحواف والزوايا** |
| **V4 neurons** | بتعالج **الأشكال الأكثر تعقيداً** |
| **AIT** | بتساعد في **التعرف على الأشياء** |

### الـ Biological Neurons:

| المكون | الوظيفة | المقابل في ANN |
|--------|---------|----------------|
| **Dendrites** | بتستقبل الإشارات من النيورونات التانية | **Inputs** |
| **Axon** | "كابل" بينقل الإشارات الكهربائية | **Links/Connections** |
| **Action Potentials** | النبضات الكهربائية اللي بتمشي في الـ axon | **Signals** |
| **Synapses** | المسافة بين النيورونات + قوة الوصلات | **Weights** |
| **Firing Rates** | معدل الإشارات — معتمد على الـ stimulus | **Activation output** |

> يعني في عقل الإنسان الـ Synapses هي المسافة بين النيورونات وقوة الوصلات اللي بينهم — ودي عندنا في الـ ANN بتمثّل الـ Weights.

---

> **نهاية المحاضرة التانية**
