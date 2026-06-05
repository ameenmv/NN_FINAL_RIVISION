# المحاضرة الرابعة — Simple Neural Network

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 124–152

---

## البنية العامة — General Architecture

الشبكة العصبية البسيطة هي **Single-layer neural network** — عبارة عن خط بيفصل المساحة لمنطقتين.

> يعني زي الـ linear regression — بيعمل خط بيفصل بيه بين الكلاسين اللي عندي. لو القيمة أكبر من 0 فكده تبع الكلاس الموجب، لو أصغر فهو كده تبع الكلاس السالب.

---

## McCulloch-Pitts Neuron (1943)

أول نموذج للنيورون الصناعي — **بسيط جداً** ومفيش فيه أي تعلم.

| الخاصية | التفاصيل |
|---------|----------|
| **الـ Output** | Binary — **0 أو 1** |
| **الـ Input** | Weighted sum of **binary inputs** |
| **قاعدة الـ Firing** | بيطلع 1 لو الـ input **≥ threshold** |
| **الـ Weights** | **Excitatory** (w > 0) أو **Inhibitory** (w = -p) |
| **التعلم** | ❌ مفيش — الـ weights والـ threshold **بيتحطوا يدوي** |
| **الاستخدام** | بوابات منطقية بسيطة: **AND, OR, NOT** |

### مثال — AND Gate:

```
Inputs: x1, x2 (binary: 0 or 1)
Weights: w1 = 1, w2 = 1
Threshold: θ = 2

Weighted Sum = w1*x1 + w2*x2

x1=0, x2=0 → 0+0 = 0 < 2 → Output: 0
x1=0, x2=1 → 0+1 = 1 < 2 → Output: 0
x1=1, x2=0 → 1+0 = 1 < 2 → Output: 0
x1=1, x2=1 → 1+1 = 2 ≥ 2 → Output: 1  ✓
```

---

## Hebb Networks — Hebbian Learning (1949)

الـ **Hebbian Learning** مبنية على فكرة بسيطة: **"Neurons that fire together, wire together"**.

### القاعدة:

```
∆w = x * y
w(new) = w(old) + ∆w
```

| الخاصية | التفاصيل |
|---------|----------|
| **قاعدة التعلم** | الـ weight بيزيد **فقط لما x و y الاتنين بيكونوا "on"** |
| **الأساس** | تقوية الرابط بين النيورونات اللي بتشتغل مع بعض |
| **الـ Output** | بيكون **0 أو 1** |

> يعني امتى الـ weight بيزيد؟ في حالة إن الـ x والـ y الاتنين بيزيدوا مع بعض (الاتنين بيكونوا بـ 1). لأن لو واحد فيهم بـ 0 فكده ضربهم هيكون بـ 0 فمفيش تغيير.

### طريقة العمل:

```
1. ابدأ بـ weights بدائية (وليكن 0)
2. لكل training example:
   - احسب ∆w = x * y
   - حدّث: w(new) = w(old) + ∆w
3. كرر على كل الـ examples
```

---

## الـ Perceptron (1957)

الـ **Perceptron** هو تطوير على McCulloch-Pitts — فيه **خوارزمية تعلم** حقيقية.

### الفرق الأساسي عن Hebb:

> هنا مش هعمل أي تعديل في الـ weight إلا لما الـ **predicted (y)** اللي أطلعه مختلف عن الـ **target (t)**. في الحالة دي بس أعمل update.

### القاعدة:

```
if y ≠ t:
    ∆w = α * (t - y) * x
    w(new) = w(old) + ∆w
    b(new) = b(old) + α * (t - y)
```

| الخاصية | التفاصيل |
|---------|----------|
| **التعلم** | **Error-driven** — بيتعلم فقط لما الـ output غلط |
| **الـ Input** | weighted sum of inputs |
| **الـ Output** | binary (0 أو 1) عن طريق step function |
| **الـ Learning Rate** | α — بتتحكم في حجم التعديل |

### امتى بيتوقف؟

- لما **مفيش أي learn جديد** (الـ weights مبتتغيرش)
- أو الـ **accuracy مش بتزيد**
- أو لما يوصل لـ **pre-determined number of epochs**

### قيود الـ Perceptron:
- بيحل **المشاكل الخطية بس** (linearly separable)
- **مش بيقدر** يحل مشكلة **XOR** — وده اللي خلّى أبحاث AI تتأخر في الـ 1960s

---

## Adaline — Adaptive Linear Neuron (1960)

الـ **Adaline** تطوير على الـ Perceptron — بيستخدم **الـ error الفعلي** (مش بس صح/غلط) في التعلم.

| الخاصية | التفاصيل |
|---------|----------|
| **الـ Output** | **Continuous value** (مش binary زي الـ Perceptron) |
| **التعلم** | مبني على تقليل **Mean Squared Error (MSE)** — يسمى **Delta Rule** |
| **طريقة الـ Update** | **Gradient Descent** |
| **الفرق عن Perceptron** | بيستخدم **الـ net input الفعلي** (قبل الـ activation) في التدريب |

### القاعدة (Delta Rule):

```
∆w = α * (t - net) * x
```

> يعني الـ Adaline بيشوف الفرق الحقيقي بين الـ target والـ net input (مش الـ output بعد الـ activation) وبيعمل update بناءً عليه.

---

## مقارنة شاملة — McCulloch-Pitts vs Hebb vs Perceptron vs Adaline

| الخاصية | McCulloch-Pitts | Hebb | Perceptron | Adaline |
|---------|----------------|------|------------|---------|
| **السنة** | 1943 | 1949 | 1957 | 1960 |
| **التعلم** | ❌ لا | ✅ نعم | ✅ نعم | ✅ نعم |
| **القاعدة** | threshold يدوي | ∆w = x*y | ∆w = α(t-y)*x | ∆w = α(t-net)*x |
| **الـ Output** | binary | binary | binary | continuous |
| **الـ Error** | — | — | (t - y) بعد activation | (t - net) قبل activation |
| **الأساس** | logic gates | correlation | error correction | gradient descent |

---

## Binary vs Multi-class Classification

### Binary Classification:
- **Target t = 0 أو 1**
- الشبكة بتطلع **قيمة واحدة** بتصنّف في واحد من كلاسين

### Multi-class Classification (K classes):
- الـ output layer فيها **K units** — كل واحدة بتمثل class
- الـ target vector بيكون **one-hot vector**:
  - مثلاً لو عندنا 4 classes (pedestrian, car, truck, bike):
    - `[1, 0, 0, 0]` → pedestrian
    - `[0, 1, 0, 0]` → car
    - `[0, 0, 1, 0]` → truck
    - `[0, 0, 0, 1]` → bike
- كل unit بتطلع **احتمال** لكل class

---

> **نهاية المحاضرة الرابعة**
