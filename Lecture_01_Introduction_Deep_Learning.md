# المحاضرة الأولى — Introduction to Deep Learning

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 1–20

---

## Traditional Programming vs Machine Learning

في البداية لازم نفهم الفرق الجوهري بين الطريقة التقليدية والـ Machine Learning:

| النموذج | الطريقة | المثال |
|---------|---------|--------|
| **Traditional Programming** | انت بتدي الكمبيوتر الـ **Data** والـ **Rules** (البرنامج)، وهو بيديك الـ **Output** | `if temperature > 30: print("Hot")` |
| **Machine Learning** | انت بتدي الكمبيوتر الـ **Data** والـ **Output** المتوقع، وهو بيتعلم الـ **Rules** لوحده | بتديله صور قطط وكلاب، وهو بيتعلم يفرق بينهم |

> يعني ببساطة: في البرمجة التقليدية انت بتكتب القواعد. في الـ ML الكمبيوتر بيكتشف القواعد لوحده من الـ data.

---

## أنواع الـ Machine Learning

### 1. Supervised Learning (تعلم بإشراف):
- الـ training data فيها **inputs + correct outputs** (labels)
- المهام الأساسية:
  - **Classification** — تصنيف (مثلاً: صورة قطة ولا كلب؟)
  - **Regression** — توقع قيمة مستمرة (مثلاً: سعر بيت)

### 2. Unsupervised Learning (تعلم بدون إشراف):
- الـ data **مفيهاش labels** — مفيش إجابة صح
- الهدف: **اكتشاف أنماط مخفية** أو تجميع البيانات المتشابهة
- مثال: تقسيم العملاء لمجموعات (Clustering)

### 3. Reinforcement Learning (تعلم بالتعزيز):
- الموديل بيتعلم عن طريق **المكافآت والعقوبات** بناءً على أفعاله
- مثال: تعليم AI يلعب لعبة — لو كسب ياخد reward، لو خسر ياخد penalty

---

## الهرم — AI vs ML vs DL

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
| **DL** | جزء من ML — بيستخدم شبكات عصبية عميقة بطبقات كتير |

---

## Shallow Learning vs Deep Learning

| الخاصية | Shallow Learning | Deep Learning |
|---------|-----------------|---------------|
| **Feature Extraction** | **يدوي** — انت بتختار الـ features بإيدك | **تلقائي** — الموديل بيتعلم الـ features لوحده |
| **الـ Classifier** | Generic classifiers زي **SVM, KNN** | شبكات عصبية عميقة بطبقات كتير |
| **الأداء** | كويس على بيانات بسيطة | **ممتاز** على بيانات معقدة (صور، نصوص، صوت) |
| **الاحتياج للبيانات** | بيانات أقل | **بيانات كتيييير** |

> يعني في الـ Shallow Learning كنا بنعمل feature extraction بإيدينا (زي ما عملنا في الترم الأول عشان نعرف ايه الـ features المهمة). لكن في الـ Deep Learning الموديل بيعمل feature extraction لوحده بدون أي تدخل مننا.

---

## تاريخ الـ Deep Learning — Brief History

| السنة | الحدث |
|-------|-------|
| **1943** | McCulloch & Pitts بيقترحوا أول نموذج لـ artificial neuron |
| **1957** | Rosenblatt بيطور الـ **Perceptron** — أول نموذج تعلم |
| **1969** | Minsky & Papert بيوضحوا حدود الـ Perceptron (**مشكلة XOR**) — أبحاث AI بتتأخر |
| **1986** | **Backpropagation** بينتشر — فتح الباب لتدريب الشبكات العميقة |
| **1995** | **SVMs** بتظهر وبتبقى شائعة في التصنيف |
| **2006** | Hinton بيقدم **Deep Belief Networks** — عصر الـ Deep Learning بيبدأ |
| **2012** | **AlexNet** بيكسب ImageNet — الـ CNNs بتنتشر |
| **2014** | **GANs** — Ian Goodfellow بيقدم الـ Generative Adversarial Networks |
| **2015** | **ResNet** — شبكات أعمق من 100 طبقة |
| **2016** | **AlphaGo** بيهزم بطل العالم في Go باستخدام Deep RL |
| **2017** | **Transformers** بيظهروا — غيّروا الـ NLP للأبد |
| **2018** | **BERT** من Google بيطوّر الـ NLP باستخدام Transfer Learning |
| **2020+** | **GPT-2, GPT-3, GPT-3.5, GPT-4** — الـ Large Language Models بتعيد تعريف الـ AI |

---

> **نهاية المحاضرة الأولى**
