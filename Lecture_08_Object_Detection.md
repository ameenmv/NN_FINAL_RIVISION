# المحاضرة التامنة — Object Detection

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 261–295

---

## مهام الـ Computer Vision

| المهمة | الوصف | مثال |
|--------|-------|------|
| **Classification** | بتديله صورة ويقولك النوع | "قطة" أو "كلب" |
| **Semantic Segmentation** | بيعرف **كل بكسل** في الصورة — بدون تمييز بين الأجسام المتشابهة | كل العشب لونه واحد |
| **Object Detection** | بيعمل **مربعات** حول كل حاجة ويحدد نوعها | مربع حول كل عربية |
| **Instance Segmentation** | بيعمل **ماسك** لكل حاجة لحد نفس النوع | كل كلب لوحده بلون مختلف |

---

## Classification + Localization

بنوسّع الـ Classification عشان نتوقع **نوع الـ object ومكانه** في الصورة.

### الـ Output:
- **Label** — نوع الـ object
- **Bounding Box** — إحداثيات (x, y, width, height)

### بيستخدم نوعين من الـ Loss:

| الـ Loss | الوظيفة |
|----------|---------|
| **Softmax Loss** | عشان يتأكد إن **التصنيف** صح |
| **L2 Loss** | عشان يضبط **مكان المربع** |

### مثال: Human Pose Estimation
بنمثل الـ pose كـ **14 نقطة مفصل** (joint positions): القدم اليمنى/الشمال، الركبة، الورك، الكتف، الكوع، اليد، الرقبة، أعلى الرأس. كل joint بيتوقع (x, y) + L2 Loss.

---

## Region Proposals

بدل ما ندور على كل الـ pixels في الصورة (sliding window)، بنحدد **المناطق اللي فيها حاجات** (زي الكتل الملونة) بسرعة.

| الطريقة | الوصف |
|---------|-------|
| **Selective Search** | خوارزمية بتحدد ~**2000** منطقة مهمة في ثواني |

---

## R-CNN (Regions with CNN)

أول محاولة ناجحة لاستخدام CNN في Object Detection — لكن **بطيء جداً**.

### الخطوات:

```
1. Generate ~2000 region proposals (Selective Search)
              ↓
2. Warp each region to fixed size (224×224)
              ↓
3. Run CNN on EACH region separately
              ↓
4. SVM for classification + Linear Regression for bounding box
```

| الخاصية | التفاصيل |
|---------|----------|
| **عدد المناطق** | ~2000 منطقة |
| **المشكلة** | **بطيء جداً** — بيشغّل CNN لكل منطقة لوحدها |
| **عدد الموديلات** | محتاج **3 موديلات منفصلة** (CNN, SVM, Regression) |

---

## Fast R-CNN

تحسين كبير على R-CNN — **أسرع بكتير**.

### الفكرة الأساسية:

بدل ما نشغّل CNN على كل منطقة لوحدها، بنشغّل CNN **مرة واحدة على الصورة كلها** ونستخرج الـ features منها.

| التحسين | التفاصيل |
|---------|----------|
| **CNN pass واحد** | للصورة كلها (مش لكل region) |
| **استبدل SVM** | بـ **Softmax** |
| **تدريب موحد** | CNN + classifier + regressor **كلهم مع بعض** |
| **ROI Pooling** | بيجمع الميزات من المناطق المختلفة بكفاءة |
| **السرعة** | تدريب **8.75 ساعة** / تنفيذ **2.3 ثانية/صورة** |

---

## Faster R-CNN

**الأسرع** — بيستبدل Selective Search بشبكة عصبية!

### الابتكار:

بيخلي الـ **CNN نفسها تتنبأ بالمناطق المهمة** — عن طريق شبكة اسمها **Region Proposal Network (RPN)**.

```
Input Image
    ↓
┌──────────┐
│   CNN    │ → Feature Map
└────┬─────┘
     ↓
┌──────────┐     ┌─────────────┐
│   RPN    │ ──→ │ Region      │
│          │     │ Proposals   │
└────┬─────┘     └──────┬──────┘
     ↓                  ↓
┌──────────────────────────────┐
│  ROI Pooling + FC + Softmax  │
│  + Bounding Box Regression   │
└──────────────────────────────┘
     ↓
  Detection Results
```

### Anchor Boxes:

**مربعات جاهزة** بأشكال ونسب مختلفة (aspect ratios) عشان تلاقي الأجسام بسهولة.

### الـ Losses:

| الـ Loss | الوظيفة |
|----------|---------|
| **Log Loss** | للتصنيف (object ولا background) |
| **Smooth L1 Loss** | لضبط إحداثيات المربع |

---

## مقارنة — R-CNN vs Fast R-CNN vs Faster R-CNN

| الخاصية | R-CNN | Fast R-CNN | Faster R-CNN |
|---------|-------|-----------|-------------|
| **Region Proposals** | Selective Search | Selective Search | **RPN (neural network)** |
| **CNN passes** | ~2000 (لكل region) | **1** (للصورة كلها) | **1** |
| **التصنيف** | SVM | Softmax | Softmax |
| **تدريب موحد؟** | ❌ 3 موديلات | ✅ | ✅ |
| **السرعة** | بطيء جداً | أسرع | **الأسرع** |

---

## YOLO — You Only Look Once

أسرع طريقة لـ Object Detection — بيشتغل في **real-time**.

### الفكرة:

```
1. بيقسم الصورة لـ Grid (مثلاً 7×7)
2. كل خلية بتتنبأ بـ:
   - Bounding boxes (عادةً 2)
   - Confidence scores
   - Class probabilities
3. كل ده في pass واحد بس!
```

| الخاصية | التفاصيل |
|---------|----------|
| **الطريقة** | Grid-based — مرة واحدة على الصورة كلها |
| **السرعة** | **سريع جداً** — ممكن يشتغل بالفيديو (real-time) |
| **العيب** | دقة أقل من Faster R-CNN على الأجسام الصغيرة |

---

## Instance Segmentation — Mask R-CNN

تطوير على **Faster R-CNN** — بيضيف **فرع في الشبكة عشان يعمل ماسك لكل جسم**.

| الخاصية | التفاصيل |
|---------|----------|
| **الـ Output** | **3 حاجات**: نوع الجسم + المربع المحيط + الصورة الدقيقة (**28×28 binary mask**) |
| **ROI Align** | بيحل مشكلة التداخل في الـ masks (أفضل من ROI Pooling) |

### RoI Pool vs RoI Align:

| الطريقة | الوصف |
|---------|-------|
| **RoI Pool** | بيعمل **snap** للـ grid cells → misalignment صغير |
| **RoI Align** | بيستخدم **bilinear interpolation** بدل الـ snapping → دقة أعلى |

---

## Semantic Segmentation — FCN

الـ **Fully Convolutional Networks** — بتصنّف **كل بكسل** في الصورة بدون تمييز بين الـ instances.

### الخطوات:

| المرحلة | الطريقة | الوظيفة |
|---------|---------|---------|
| **Down Sampling** | Pooling / Strided Conv | تقليل دقة الصورة عشان السرعة |
| **Up Sampling** | Transpose Convolution | يزوّد الدقة تاني |
| **Skip Connections** | دمج التفاصيل المهمة مع باقي التفاصيل | يحافظ على الدقة |

### طرق الـ Upsampling:

| الطريقة | الوصف | الدقة |
|---------|-------|-------|
| **Nearest Neighbor** | بيكرر قيم الـ pixels (بسيط وسريع) | ⭐ |
| **Max Unpooling** | بيعكس الـ max pooling باستخدام الأماكن اللي اتذكرت | ⭐⭐ |
| **Transpose Convolution** | بيزوّد الدقة بطريقة ذكية وقابلة للتعلم (الأدق) | ⭐⭐⭐ |

### FCN Variants (Skip Connections):

| النوع | الوصف | الدقة |
|------|-------|-------|
| **FCN-32s** | 32x upsampled prediction من الـ pool5 | الأقل دقة |
| **FCN-16s** | دمج pool4 + 2x upsampled pool5 ثم 16x upsample | أحسن |
| **FCN-8s** | دمج pool3 + pool4 + pool5 ثم 8x upsample | **الأدق** |

---

## ملخص مهام الـ Computer Vision

```
Classification:          Object Detection:       Semantic Seg:          Instance Seg:
┌──────────┐            ┌──────────┐            ┌──────────┐          ┌──────────┐
│          │            │ ┌──┐     │            │▓▓▓▓░░░░░░│          │▓▓▓▓░░░░░░│
│   Cat    │            │ │  │ Cat │            │▓▓▓▓░░░░░░│          │▓▓▓▓░░░░░░│
│          │            │ └──┘     │            │▓▓▓▓░░░░░░│          │████░░░░░░│
│          │            │    ┌───┐ │            │░░░░░░░░░░│          │░░░░░░░░░░│
│          │            │    │Dog│ │            │░░░░░░░░░░│          │░░░░░░░░░░│
└──────────┘            └────┴───┘─┘            └──────────┘          └──────────┘
Label: Cat              Labels + Boxes          Every pixel           Each object
                                                labeled               unique mask
```

---

> **نهاية المحاضرة التامنة**
