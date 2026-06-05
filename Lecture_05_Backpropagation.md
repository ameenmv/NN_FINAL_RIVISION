# المحاضرة الخامسة — Backpropagation Algorithm

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 153–186

---

## الـ Backpropagation — إيه هو؟

الـ **Backpropagation** هو الخوارزمية الأساسية لتدريب الشبكات العصبية — بيحسب الـ **gradient** بتاع الـ loss function بالنسبة للـ weights باستخدام **Gradient Descent**.

> يعني بيعرّفنا الـ weight اللي بتسبب الخطأ ونسبته قد ايه عشان نعدّلها.

### الخطوتين الأساسيتين:

| الخطوة | الاسم | الوصف |
|--------|-------|-------|
| **1** | **Forward Step** | بنمشي الـ activation من الـ **input layer** للـ **output layer** — بنحسب الـ output |
| **2** | **Backward Step** | بنمشي الـ errors من الـ **output layer** للـ **hidden layers** — بنحسب الـ gradients وبنعدّل الـ weights |

```
Forward:   Input → Hidden₁ → Hidden₂ → ... → Output
                                                  ↓
                                              Compute Loss
                                                  ↓
Backward:  Input ← Hidden₁ ← Hidden₂ ← ... ← Output
           (update weights using gradients)
```

---

## الـ Chain Rule

الـ **Chain Rule** بنستخدمها لما نحاول نشتق متغير عشان نشوف **تأثيره على متغير تاني** — يعني لما الـ function جوا function تانية.

```
dL/dw = dL/da * da/dz * dz/dw
```

> يعني لو عايز أعرف تأثير الـ weight على الـ loss، بفكّك المسار لخطوات صغيرة وبضربهم في بعض.

### مثال بسيط:

```
z = w*x + b        ← الـ weighted sum
a = σ(z)            ← الـ activation (sigmoid)
L = -(y*log(a) + (1-y)*log(1-a))   ← الـ loss

dL/dw = dL/da * da/dz * dz/dw
      = (a - y) * x
```

---

## الـ Momentum

الـ **Momentum** هو method بيساعد الـ SGD إنه يتسارع في الاتجاه الصح ويقلل التذبذبات.

### الفكرة:

```
v(t+1) = β * v(t) + α * ∂L/∂w
w(t+1) = w(t) - v(t+1)
```

| الرمز | المعنى |
|-------|--------|
| **v** | الـ velocity (سرعة التحديث) |
| **β** | معامل الـ momentum (عادةً 0.9) |
| **α** | الـ learning rate |

### الميزة:
- التحديث للـ weight الحالي بيحتوي على **momentum من التحديثات السابقة**
- بيمنع التغيير المفاجئ في اتجاه الـ weight update (**smoothing the learning process**)

> يعني إضافة الـ Momentum معناها إن التحديث بتاع الـ weight الحالي بيعتمد على الـ weight السابق — وده بيدي سرعة أكبر في نفس اتجاه التحرك وسلاسة.

---

## الـ Learning Rate — مشاكل وحلول

| الحالة | النتيجة |
|--------|---------|
| **Learning rate كبير** | الـ Cost ممكن **متقلش** بعد كل update — بيعمل overshoot |
| **Learning rate صغير** | التدريب هيبقى **بطيء جداً** |

---

## Learning Rate Schedules

الـ **Learning Rate Schedules** بتحسّن التدريب عن طريق **تعديل الـ learning rate أثناء التدريب**.

> الفكرة: الموديل يبدأ بـ learning rate **عالي** (يتعلم بسرعة) وبعدين يقلله تدريجياً (للـ fine-tuning).

### الأنواع:

| النوع | الطريقة |
|-------|---------|
| **Time-based Decay** | الـ learning rate بيقل مع الوقت — في كل epoch بيقل شوية |
| **Step Decay** | بينزّل الـ learning rate **فجأة** بنسبة معينة كل كام epoch |
| **Exponential Decay** | بيضرب الـ learning rate في رقم أقل من 1 (مثلاً 0.96) بعد كل step |
| **Adaptive Methods** | خوارزميات زي **Adam** أو **RMSprop** بتعدّل الـ learning rate تلقائياً لكل parameter |

> غالباً بتُستخدم مع **Stochastic Gradient Descent (SGD)**.

---

## Adagrad — Adaptive Gradient

| الخاصية | التفاصيل |
|---------|----------|
| **الفكرة** | **Derivatives كبيرة** → Learning Rate **أصغر** / **Derivatives صغيرة** → Learning Rate **أكبر** |
| **الميزة** | بيعمل adapt للـ learning rate لكل parameter |
| **المشكلة** | مش كويس لما الـ **objective non-convex** (اللي فيها قمم وقيعان) |
| **ليه؟** | لأنه بيجمع كل الـ squared gradients من أول التدريب — الـ learning rate بيقل لحد ما يتوقف عن التعلم |

> يعني الـ Adagrad مش بيعرف يتعامل مع الـ non-convex objectives — بيقف عند **local minimum** ومش بيقدر يوصل للـ **global minimum**.

---

## RMSprop

الـ **RMSprop** بيحل مشكلة الـ Adagrad — بدل ما يجمّع كل squared gradients من أول التدريب، بيحسب **متوسط متحرك (moving average)** للـ squared gradients الأخيرة بس.

```
v(t) = β * v(t-1) + (1-β) * (∂L/∂w)²
w(t) = w(t-1) - α / √(v(t) + ε) * ∂L/∂w
```

### المميزات:

| الميزة | التفاصيل |
|--------|----------|
| **Fast Convergence** | بيتعلم أسرع من optimizers كتير |
| **Stable Learning** | بيتجنب التحديثات الكبيرة وبيخلي التدريب مستقر |
| **Good for Non-convex** | بيشتغل كويس لما يكون فيه **local minima** كتير (زي deep learning) — بيتخلص من الـ local minimum |

> يعني بدل ما يجمّع كل الـ gradients من أول التدريب زي Adagrad، بياخد الجديد وينسى القديم.

---

## Adam — Adaptive Moment Estimation

الـ **Adam** هو أشهر وأقوى optimizer — بيجمع مميزات **Momentum** و **RMSprop** مع بعض.

```
m(t) = β₁ * m(t-1) + (1-β₁) * ∂L/∂w          ← 1st moment (mean) — زي Momentum
v(t) = β₂ * v(t-1) + (1-β₂) * (∂L/∂w)²       ← 2nd moment (variance) — زي RMSprop

m̂(t) = m(t) / (1 - β₁ᵗ)                       ← bias correction
v̂(t) = v(t) / (1 - β₂ᵗ)                       ← bias correction

w(t) = w(t-1) - α * m̂(t) / (√v̂(t) + ε)
```

| الرمز | القيمة الافتراضية | المعنى |
|-------|-------------------|--------|
| **β₁** | 0.9 | decay rate للـ 1st moment |
| **β₂** | 0.999 | decay rate للـ 2nd moment |
| **ε** | 10⁻⁸ | عشان نتجنب القسمة على صفر |
| **α** | 0.001 | الـ learning rate |

---

## مقارنة الـ Optimizers

| الـ Optimizer | الفكرة | المميزات | العيوب |
|-------------|--------|----------|--------|
| **SGD** | basic gradient descent | بسيط | بطيء، بيتذبذب |
| **SGD + Momentum** | بيضيف velocity | أسرع، أسلس | محتاج tuning |
| **Adagrad** | adaptive per-parameter LR | كويس للـ sparse data | LR بيقل كتير |
| **RMSprop** | moving average of squared grads | مستقر، كويس لـ non-convex | — |
| **Adam** | Momentum + RMSprop | الأفضل عموماً | ممكن يحتاج tuning |

---

## الـ Dropout — تقنية منع الـ Overfitting

الـ **Dropout** هو تقنية **regularization** بنستخدمها عشان نقلل الـ **overfitting** في الشبكات العصبية.

### الفكرة:

بيروح **يشيل بعض النيورونات بشكل عشوائي** (يخليها 0) سواء اللي في الـ input أو الـ hidden layers أثناء التدريب — وكمان يشيل كل الروابط (forward و backward) المتعلقة بيهم.

```
بدون Dropout:          مع Dropout (p=0.5):
  o   o   o              o       o
 /|\ /|\ /|\            / \     /|\
o   o   o   o          o   o       o     ← 50% من النيورونات اتشالوا
 \|/ \|/ \|/              \ /       |
  o   o   o                o       o
```

### تفاصيل:

| الخاصية | التفاصيل |
|---------|----------|
| **Dropout Probability (p)** | نسبة النيورونات اللي هتتشال — مثلاً `p=0.5` يعني 50% هيتشالوا |
| **في التدريب** | كل mini-batch بنشيل مجموعة عشوائية جديدة — يعني هيكل الشبكة **بيتغير كل دورة** |
| **تأثير على الـ Weights** | الـ weights بتتحدث **بس للنيورونات اللي مش اتشالوا** |
| **في الـ Testing** | **مبنعملش أي dropout خالص** — لكن الـ weights بتتضرب في **(1-p)** عشان نعوّض النيورونات اللي اتشالت في التدريب |

### الفوايد:

- بيمنع الـ **overfitting**
- بيحسّن **generalization** — الموديل بيشتغل أحسن على data جديدة مشافهاش قبل كده

---

> **نهاية المحاضرة الخامسة**
