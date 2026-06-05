# المحاضرة التالتة — Linear Regression

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 64–123

---

## مشكلة الـ Regression

الـ **Regression** هي مشكلة **Supervised Learning** — يعني الموديل بيتدرب على بيانات فيها الإجابة الصح (labeled data).

**الهدف**: توقع **قيمة مستمرة (continuous)** — مش تصنيف.

> يعني مثلاً: عايز أتوقع **سعر بيت** بناءً على مساحته. الـ output هنا رقم مش class.

### مثال — Housing Prices:
- **Input (x)**: مساحة البيت
- **Output (y)**: سعر البيت
- الموديل بيتعلم العلاقة بين x و y

---

## الـ Linear Regression

الـ **Linear Regression** هو أبسط أنواع الـ regression — بيرسم **خط مستقيم** يمثّل العلاقة بين الـ input والـ output.

### الصيغة:

```
h(x) = θ₀ + θ₁ * x
```

| الرمز | المعنى |
|-------|--------|
| **h(x)** | الـ hypothesis — القيمة المتوقعة |
| **θ₀** | الـ intercept (الـ bias) |
| **θ₁** | الـ slope (الميل) |
| **x** | الـ input feature |

> لو عندنا variable واحد بس بنسميه **Univariate Linear Regression**.

---

## الـ Cost Function (Loss Function)

الـ **Cost Function** بتقيس **الخطأ** بين القيم المتوقعة (predicted) والقيم الفعلية (actual). الهدف إننا **نقلل** الـ cost function أثناء التدريب.

### Mean Squared Error (MSE):

```
J(θ₀, θ₁) = (1/2m) * Σ(h(xⁱ) - yⁱ)²
```

| الرمز | المعنى |
|-------|--------|
| **J** | الـ cost function |
| **m** | عدد الـ training examples |
| **h(xⁱ)** | القيمة المتوقعة للمثال i |
| **yⁱ** | القيمة الحقيقية للمثال i |

> يعني ببساطة: بناخد الفرق بين المتوقع والحقيقي، بنربّعه، بنجمع كل الفروق، وبنقسم على 2m.

---

## الـ Gradient Descent

الـ **Gradient Descent** هو خوارزمية **تحسين تكرارية (iterative optimization)** — بنستخدمها عشان نلاقي أقل قيمة للـ cost function.

### الفكرة:

```
θⱼ := θⱼ - α * ∂J/∂θⱼ
```

| الرمز | المعنى |
|-------|--------|
| **α** | **Learning Rate** — حجم الخطوة في كل update |
| **∂J/∂θⱼ** | **الاشتقاق** — اتجاه الانحدار |

### إزاي بيشتغل:

```
1. ابدأ بقيم عشوائية لـ θ
2. احسب الـ gradient (الاشتقاق) بتاع الـ cost function
3. حرّك الـ parameters في الاتجاه المعاكس للـ gradient
4. كرر لحد ما توصل للـ convergence (أقل قيمة)
```

### الـ Gradient للـ Linear Regression:

```
تكرار لحد الـ convergence:
  θ₀ := θ₀ - α * (1/m) * Σ(h(xⁱ) - yⁱ)
  θ₁ := θ₁ - α * (1/m) * Σ(h(xⁱ) - yⁱ) * xⁱ
  (لازم يتعملوا simultaneously update!)
```

> ❗ **مهم:** لازم نعمل update لكل الـ parameters **في نفس الوقت** (بنحسب القيم الجديدة كلها الأول، وبعدين نعمل assign). لو عملنا update لـ θ₀ الأول وبعدين استخدمناه في حساب θ₁ — النتيجة هتبقى **غلط**.

---

## الـ Learning Rate (α)

الـ **Learning Rate** هو عامل بيحدد **حجم التغييرات** اللي بيتم إجراؤها في الـ parameters في كل خطوة.

| الحالة | النتيجة |
|--------|---------|
| **α صغير أوي** | التحديثات هتبقى **بطيئة جداً** — التدريب هياخد وقت طويل |
| **α كبير أوي** | ممكن **يتخطى الـ minimum** ويعمل overshoot — الـ cost مش هتقل |

```
          Cost
           |
  كبير ← |  ╱╲  ╱╲  ╱╲    ← بيتخطى الـ minimum (overshoot)
    α    |╱    ╲╱    ╲╱
           |
           |
  مثالي ← | ╲
    α    |   ╲
           |     ╲___        ← بيوصل للـ minimum بسلاسة
           |
  صغير ← | ╲
    α    |   ╲
           |     ╲
           |       ╲
           |         ╲___   ← بيوصل بس بطيء جداً
```

---

## Feature Scaling (Normalization)

الـ **Feature Scaling** بيعمل تعديل على قيم الـ features عشان تكون في **نفس النطاق** — وده بيسرّع الـ Gradient Descent.

### الصيغة:

```
x_normalized = (x - x_min) / (x_max - x_min)
```

> بيحوّل القيم لنطاق بين **0 و 1** (أو **-1 و 1**).

### ليه مهم؟
لو feature واحد قيمته من 0 لـ 2000 وfeature تاني من 0 لـ 5، الـ gradient descent هيبقى بطيء وغير متوازن. الـ scaling بيخلي كل الـ features في نفس المستوى.

### طرق الـ Normalization:

```
1. Min-Max Normalization (0 لـ 1):    x' = (x - min) / (max - min)
2. Min-Max Normalization (-1 لـ 1):   x' = 2 * (x - min) / (max - min) - 1
3. Z-Score (Mean Normalization):       x' = (x - μ) / σ
   حيث μ = المتوسط، σ = الانحراف المعياري
```

---

## Multivariate Linear Regression

لما يكون عندنا **أكتر من feature** (مثلاً: مساحة، عدد غرف، عدد أدوار، عمر البيت):

```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

### الـ Gradient Descent للـ Multivariate:

```
تكرار لحد الـ convergence:
  θⱼ := θⱼ - α * (1/m) * Σ(h(xⁱ) - yⁱ) * xⱼⁱ
  (لكل j من 0 لـ n — simultaneously update)
```

> يعني ببساطة: نفس الفكرة بالظبط بس بنعملها لكل feature (x₁, x₂, ..., xₙ).

---

## الـ Logistic Regression

الـ **Logistic Regression** مش regression فعلياً — هو **classification** algorithm. الهدف إنه يتوقع **احتمال** بين 0 و 1.

### الصيغة:

```
h(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

> هي هي الـ **Sigmoid** اللي شرحناها قبل كده.

### الـ Decision Boundary:

الـ **Decision Boundary** هو **hyperplane** بيقسم المساحة لمنطقتين:
- لو `h(x) ≥ 0.5` → **Class 1** (الكلاس الموجب)
- لو `h(x) < 0.5` → **Class 0** (الكلاس السالب)

| النوع | الوصف |
|-------|-------|
| **Linear Decision Boundary** | خط مستقيم (أو hyperplane) بيفصل بين الكلاسات |
| **Non-linear Decision Boundary** | منحنى أو شكل معقد — بنحتاج polynomial features |

> يعني لو الـ data مش ممكن تتفصل بخط مستقيم، لازم نستخدم features أعقد (polynomial) عشان الـ boundary يبقى منحنى.

---

## Logistic Regression Cost Function

الـ cost function بتاعة الـ Logistic Regression مختلفة عن الـ Linear Regression — بتستخدم **Log Loss (Cross-Entropy)**:

```
J(θ) = -(1/m) * Σ[ yⁱ * log(h(xⁱ)) + (1 - yⁱ) * log(1 - h(xⁱ)) ]
```

### ليه مش MSE؟
لأن لو استخدمنا MSE مع Sigmoid، الـ cost function هتبقى **non-convex** (فيها local minima كتير) — الـ gradient descent مش هيوصل للـ global minimum.

الـ **Log Loss** بتضمن إن الـ cost function تكون **convex** — يعني ليها minimum واحد بس.

### Gradient Descent للـ Logistic Regression:

```
θⱼ := θⱼ - α * (1/m) * Σ(h(xⁱ) - yⁱ) * xⱼⁱ
```

> ❗ الصيغة شكلها **زي الـ Linear Regression بالظبط**، بس الفرق إن h(x) هنا = σ(θᵀx) مش θᵀx. ده بيطلع من اشتقاق الـ Log Loss.

---

## ملخص المقارنة — Linear vs Logistic Regression

| الخاصية | Linear Regression | Logistic Regression |
|---------|-------------------|---------------------|
| **النوع** | Regression (توقع قيمة) | Classification (تصنيف) |
| **الـ Output** | أي رقم (continuous) | احتمال بين 0 و 1 |
| **الـ Function** | `h(x) = θ₀ + θ₁x` | `h(x) = σ(θᵀx)` |
| **الـ Cost** | MSE | Log Loss (Cross-Entropy) |
| **الـ Decision** | خط أفضل fit | boundary يفصل الكلاسات |

---

> **نهاية المحاضرة التالتة**
