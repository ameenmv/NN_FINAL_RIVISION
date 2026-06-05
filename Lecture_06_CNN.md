# المحاضرة السادسة — Convolutional Neural Networks (CNN)

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 187–235

---

## ليه CNN؟

الشبكات العادية (Fully Connected) مش عملية للصور — لو صورة 200×200×3 يبقى عندنا **120,000 weight** لنيورون واحد بس! ده كتير جداً ومش efficient.

الـ **CNN** بتحل المشكلة دي عن طريق إنها **بتستغل الـ spatial structure** بتاع الصورة — بدل ما كل نيورون يبص على كل pixel، كل نيورون بيبص على **جزء صغير** (local region) من الصورة.

> يعني بدل ما الشبكة تشوف الصورة كلها مرة واحدة، بتشوفها حتة حتة وتستخرج الـ features المهمة.

---

## بنية الـ CNN

```
Input Image → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

```
         Input          Conv         Pool          Conv         Pool        FC       Output
      ┌─────────┐   ┌─────────┐  ┌───────┐   ┌─────────┐  ┌───────┐  ┌──────┐
      │  Image  │──→│ Filters │──→│Reduce │──→│ Filters │──→│Reduce │──→│ Dense│──→ Class
      │ 32×32×3 │   │Features │  │ Size  │   │Features │  │ Size  │  │Layers│
      └─────────┘   └─────────┘  └───────┘   └─────────┘  └───────┘  └──────┘
```

الـ CNN بتتكون من 3 أنواع رئيسية من الطبقات:

---

## 1. Convolution Layer

الـ **Convolution Layer** هي الطبقة الأساسية في الـ CNN — بتستخدم **فلاتر (filters/kernels)** بتمشي على الصورة عشان تستخرج features.

### إزاي بتشتغل:

```
Input (5×5)              Filter (3×3)           Output (3×3)
┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │   │ 1 │ 0 │ 1 │         ┌───┬───┬───┐
├───┼───┼───┼───┼───┤   ├───┼───┼───┤         │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │   │ 0 │ 1 │ 0 │         ├───┼───┼───┤
├───┼───┼───┼───┼───┤   ├───┼───┼───┤         │ 2 │ 4 │ 3 │
│ 1 │ 0 │ 1 │ 0 │ 1 │   │ 1 │ 0 │ 1 │         ├───┼───┼───┤
├───┼───┼───┼───┼───┤   └───┴───┴───┘         │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │   Element-wise           └───┴───┴───┘
├───┼───┼───┼───┼───┤   multiply + sum         Feature Map
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘
```

> الفلتر بيتحرك على الصورة، في كل موقع بيضرب element-wise وبيجمع — والنتيجة هي الـ **Feature Map**.

### الـ Hyperparameters:

| الـ Parameter | الرمز | الوظيفة |
|-------------|-------|---------|
| **Filter Size** | F | حجم الفلتر (مثلاً 3×3 أو 5×5) |
| **Stride** | S | عدد الـ pixels اللي الفلتر بيتحرك بيها في كل خطوة |
| **Padding** | P | إضافة أصفار حول الصورة عشان نتحكم في حجم الـ output |
| **Number of Filters** | K | عدد الفلاتر — كل واحد بيستخرج feature مختلف |

### حساب حجم الـ Output:

```
Output Size = (N - F + 2P) / S + 1
```

| الرمز | المعنى |
|-------|--------|
| **N** | حجم الـ input |
| **F** | حجم الـ filter |
| **P** | الـ padding |
| **S** | الـ stride |

### مثال:

```
Input: 32×32
Filter: 5×5, Stride: 1, Padding: 0

Output = (32 - 5 + 0) / 1 + 1 = 28×28
```

### أنواع الـ Padding:

| النوع | الوصف | النتيجة |
|-------|-------|---------|
| **Valid (No Padding)** | مفيش padding — الـ output أصغر | Output < Input |
| **Same Padding** | بنضيف padding عشان الـ output = الـ input | Output = Input |

---

## 2. Pooling Layer

الـ **Pooling Layer** بتقلل حجم الـ feature map بدون فقد المعلومات المهمة — بتقلل الـ **spatial dimensions** وبالتالي بتقلل عدد الـ parameters والـ computation.

### الأنواع:

| النوع | الطريقة | الاستخدام |
|-------|---------|-----------|
| **Max Pooling** | بياخد **أكبر قيمة** من كل region | الأكثر شيوعاً — بيحافظ على الـ features الأقوى |
| **Average Pooling** | بياخد **المتوسط** من كل region | بيحافظ على المعلومات العامة |

### مثال — Max Pooling (2×2, stride 2):

```
Input (4×4)                    Output (2×2)
┌───┬───┬───┬───┐             ┌───┬───┐
│ 1 │ 3 │ 2 │ 1 │             │ 4 │ 6 │   max(1,3,0,4)=4
├───┼───┼───┼───┤      →      ├───┼───┤   max(2,1,5,6)=6
│ 0 │ 4 │ 5 │ 6 │             │ 8 │ 9 │
├───┼───┼───┼───┤             └───┴───┘
│ 7 │ 8 │ 1 │ 0 │
├───┼───┼───┼───┤
│ 2 │ 3 │ 9 │ 4 │
└───┴───┴───┴───┘
```

> الـ pooling بيقلل حجم الصورة وبيخلي الشبكة أسرع، وفي نفس الوقت بيحافظ على أهم الـ features.

---

## 3. Fully Connected Layer

بعد ما الـ Conv و Pooling layers يستخرجوا الـ features، بنعمل **flatten** للـ feature maps (بنحوّلها لـ vector واحد طويل) وبندخلها على **Fully Connected layers** عادية عشان تعمل التصنيف النهائي.

```
Feature Maps → Flatten → [FC₁ → ReLU → FC₂ → ReLU → FC₃ → Softmax] → Output
```

---

## الـ Feature Hierarchy

كل ما نروح أعمق في الشبكة، الـ features بتبقى **أعقد**:

```
Layer 1: بتتعلم Edges (حواف) وخطوط بسيطة
         ↓
Layer 2: بتتعلم Textures (أنسجة) وأنماط صغيرة
         ↓
Layer 3: بتتعلم Parts (أجزاء) — عين، أنف، عجلة
         ↓
Layer 4: بتتعلم Objects (أشياء كاملة) — وش، عربية
```

> يعني الطبقات الأولى بتشوف حاجات بسيطة (خطوط وحواف)، وكل ما نروح أعمق بتفهم حاجات أكبر وأعقد.

---

## CNN vs Fully Connected Network

| الخاصية | Fully Connected | CNN |
|---------|----------------|-----|
| **الـ Connection** | كل نيورون متوصل بكل input | كل نيورون بيبص على **local region** بس |
| **الـ Parameters** | كتير جداً | أقل بكتير (weight sharing) |
| **الـ Spatial Info** | بيضيعها | **بيحافظ عليها** |
| **الاستخدام** | بيانات عادية (tabular) | **صور، فيديو، NLP** |
| **الـ Translation Invariance** | مفيش | ✅ بيتعرف على الـ feature في أي مكان |

---

## مفاهيم مهمة

### Weight Sharing:
نفس الفلتر (نفس الـ weights) بيتحرك على **الصورة كلها** — مش بنستخدم weights مختلفة لكل موقع. ده بيقلل عدد الـ parameters بشكل كبير.

### Translation Invariance:
الشبكة بتتعرف على الـ feature (مثلاً وش) **بغض النظر عن مكانه** في الصورة — ده بسبب الـ weight sharing.

### Depth:
عدد الـ filters في الـ Conv layer بيحدد **عمق** الـ output — لو استخدمنا 32 filter، الـ output هيكون بعمق 32.

---

## ملخص — Pipeline كامل لـ CNN

```
Input Image (32×32×3)
        ↓
[Conv (5×5, 6 filters) → ReLU]  → 28×28×6
        ↓
[Max Pool (2×2)]                 → 14×14×6
        ↓
[Conv (5×5, 16 filters) → ReLU] → 10×10×16
        ↓
[Max Pool (2×2)]                 → 5×5×16
        ↓
[Flatten]                        → 400
        ↓
[FC (400→120) → ReLU]           → 120
        ↓
[FC (120→84) → ReLU]            → 84
        ↓
[FC (84→10) → Softmax]          → 10 classes
```

---

> **نهاية المحاضرة السادسة**
