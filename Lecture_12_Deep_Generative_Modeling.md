# المحاضرة الثانية عشر — Deep Generative Modeling

> Neural Network Course — FCIS, Mansoura University
> Book Pages: 380–415

---

## Generative vs Discriminative Models

| النوع | الوظيفة | مثال |
|-------|---------|------|
| **Discriminative** | بيتعلم يفرق **بين الكلاسات** — بيتعلم الـ boundary | Classification (قطة ولا كلب؟) |
| **Generative** | بيتعلم **توزيع البيانات نفسها** — بيقدر يولّد بيانات **جديدة** | توليد صور وجوه جديدة |

> يعني الـ Discriminative بيجاوب على "ده إيه؟" — لكن الـ Generative بيقدر يعمل حاجة **جديدة من الأول**.

---

## Autoencoders

الـ **Autoencoder** شبكة بتتعلم **تضغط** البيانات وبعدين **تفكها** تاني — الهدف إن الـ output يطلع **قريب من الـ input**.

### البنية:

```
Input (x)          Latent Space (z)          Output (x̂)
 ┌──────┐          ┌──────┐                 ┌──────┐
 │      │ Encoder  │      │    Decoder      │      │
 │ 784  │ ──────→  │  32  │  ──────→        │ 784  │
 │      │ (compress)│     │  (reconstruct)  │      │
 └──────┘          └──────┘                 └──────┘
                      ↑
                 Bottleneck
           (compressed representation)
```

| المكون | الوظيفة |
|--------|---------|
| **Encoder** | بيضغط الـ input لـ **representation أصغر** (latent space) |
| **Latent Space (z)** | التمثيل المضغوط — الـ **bottleneck** |
| **Decoder** | بيعيد بناء الـ input من الـ latent space |

### الـ Loss Function:

```
L = ||x - x̂||²    ← Reconstruction Loss (MSE)
```

> الهدف: نقلل الفرق بين الـ input الأصلي والـ output المعاد بناؤه.

### الاستخدامات:
- **Dimensionality Reduction** — بديل لـ PCA
- **Denoising** — شيل الـ noise من الصور
- **Feature Learning** — تعلم representations مفيدة

---

## Variational Autoencoders (VAE)

الـ **VAE** تطوير على الـ Autoencoder العادي — بدل ما الـ encoder يطلع **نقطة واحدة** في الـ latent space، بيطلع **توزيع احتمالي** (mean + variance).

### الفرق عن الـ Autoencoder العادي:

| الخاصية | Autoencoder | VAE |
|---------|-------------|-----|
| **الـ Latent Space** | نقطة ثابتة (z) | **توزيع** (μ, σ²) |
| **الـ Sampling** | مفيش | بنعمل **sample** من التوزيع |
| **الـ Generation** | مش كويس في التوليد | **كويس** في توليد بيانات جديدة |
| **الـ Loss** | Reconstruction loss بس | Reconstruction + **KL Divergence** |

### البنية:

```
Input (x)
    ↓
 Encoder
    ↓
 μ (mean) + σ² (variance)     ← بيطلع توزيع مش نقطة
    ↓
 z = μ + σ × ε                ← Reparameterization Trick (ε ~ N(0,1))
    ↓
 Decoder
    ↓
Output (x̂)
```

### الـ Loss Function:

```
L = Reconstruction Loss + KL Divergence
  = ||x - x̂||² + KL(q(z|x) || p(z))
```

| الجزء | الوظيفة |
|-------|---------|
| **Reconstruction Loss** | الـ output يطلع قريب من الـ input |
| **KL Divergence** | الـ latent distribution يبقى قريب من Normal Distribution |

> الـ Reparameterization Trick عشان نقدر نعمل backpropagation عبر عملية الـ sampling — بنحوّلها لعملية حسابية.

---

## Generative Adversarial Networks (GANs)

الـ **GAN** مكون من **شبكتين بتتنافسا مع بعض**:

```
Random Noise (z)
      ↓
┌─────────────┐
│  GENERATOR  │ → Fake Data (x̂)
│   (G)       │         ↓
└─────────────┘    ┌──────────────┐      Real/Fake?
                   │DISCRIMINATOR │ ──→   (0 or 1)
Real Data (x) ──→  │    (D)       │
                   └──────────────┘
```

| المكون | الوظيفة |
|--------|---------|
| **Generator (G)** | بياخد **random noise** وبيحوّله لـ **data مزيفة** تبان حقيقية |
| **Discriminator (D)** | بيحاول **يفرق** بين الـ data الحقيقية والمزيفة |

### الـ Training — Min-Max Game:

```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

| المرحلة | الوصف |
|---------|-------|
| **تدريب الـ D** | بنثبّت G وبندرب D إنه **يفرق** أحسن بين الحقيقي والمزيف |
| **تدريب الـ G** | بنثبّت D وبندرب G إنه **يخدع** الـ D — يولّد data أحسن |
| **التوازن** | اللعبة بتستمر لحد ما الـ G يبقى كويس لدرجة إن الـ D مش بيقدر يفرق |

> يعني الـ Generator والـ Discriminator بيتنافسوا — الـ Generator بيحاول يعمل صور مزيفة مقنعة، والـ Discriminator بيحاول يكشفها. مع الوقت الاتنين بيتحسنوا.

### مشاكل الـ GANs:

| المشكلة | الوصف |
|---------|-------|
| **Mode Collapse** | الـ Generator بيتعلم يولّد **نوع واحد بس** من الـ data بدل التنوع |
| **Training Instability** | التدريب ممكن يكون **مش مستقر** — الـ G والـ D ممكن ميتوازنوش |
| **Vanishing Gradient** | لو الـ D بقى كويس أوي بسرعة، الـ G مش هيتعلم |

---

## تطبيقات الـ Generative Models

| التطبيق | الوصف | الموديل |
|---------|-------|---------|
| **Image Generation** | توليد صور وجوه واقعية مش موجودة | GAN (StyleGAN) |
| **Image-to-Image Translation** | تحويل sketch لصورة حقيقية | Pix2Pix, CycleGAN |
| **Super Resolution** | تكبير صورة صغيرة بدون فقد الجودة | SRGAN |
| **Style Transfer** | نقل أسلوب صورة لصورة تانية | Neural Style Transfer |
| **Text-to-Image** | توليد صورة من وصف نصي | DALL-E, Stable Diffusion |
| **Data Augmentation** | توليد بيانات تدريب إضافية | GAN |
| **Anomaly Detection** | اكتشاف البيانات الغريبة | Autoencoder, VAE |

---

## مقارنة — Autoencoder vs VAE vs GAN

| الخاصية | Autoencoder | VAE | GAN |
|---------|-------------|-----|-----|
| **الهدف** | إعادة بناء الـ input | توليد بيانات جديدة | توليد بيانات واقعية |
| **الـ Latent Space** | نقطة ثابتة | توزيع احتمالي | random noise |
| **الـ Loss** | Reconstruction | Reconstruction + KL | Adversarial (min-max) |
| **جودة التوليد** | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **سهولة التدريب** | ⭐⭐⭐ | ⭐⭐ | ⭐ (صعب) |
| **التنوع** | — | ⭐⭐⭐ | ⭐⭐ (mode collapse) |

---

> **نهاية المحاضرة الثانية عشر**
