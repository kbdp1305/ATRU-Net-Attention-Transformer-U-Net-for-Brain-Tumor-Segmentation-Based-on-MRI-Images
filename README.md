# 🧠 ATRU-Net: Attention Transformer U-Net for Brain Tumor Segmentation Based on MRI Images  
### 🥉 **3rd Place Winner — GEMASTIK XVII (2024)**  
### 📖 *Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi (GEMASTIK)*  
**Officially organized by Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi Republik Indonesia (KEMENRISTEKDIKTI)**  
🏅 *ICT Scientific Paper Division*  👥 *Team Rubik Riset SC — Universitas Gadjah Mada*  
**Krisna Bayu Dharma Putra · Rabbani Nur Kumoro · Vincent Yeozekiel · Dzikri Rahadian Fudholi**

---

## 📘 Overview
**ATRU-Net (Attention Transformer U-Net)** is a deep learning model designed to perform **brain tumor segmentation** on MRI scans.  
Unlike traditional U-Net architectures, ATRU-Net introduces **three complementary modules**:  
1. **Feature Pyramid Network (FPN)** for multi-scale feature recovery  
2. **Global Spatial Attention (GSA)** for boundary precision and region refinement  
3. **Transformer Encoder** for global spatial dependency learning  

By combining convolutional and transformer-based mechanisms, ATRU-Net enhances contextual awareness while maintaining fine spatial details, achieving higher **Dice (0.8827)** and **IoU (0.8292)** scores compared to U-Net baselines.  
This innovation earned the **3rd place nationally** at **GEMASTIK XVII 2024**, among **386 competing teams** in the *ICT Scientific Paper Division*.

---

## 🧭 Motivation
Brain tumor segmentation is a fundamental step in neuroimaging analysis. Manual segmentation is laborious and inconsistent across clinicians.  
Classical CNNs (including U-Net) are effective in extracting local details but fail to model **global context** and often lose **semantic consistency** due to pooling and convolution locality.  

To solve this, **ATRU-Net** integrates:  
- **Multi-scale fusion (FPN)** for semantic feature recovery  
- **Transformer-based self-attention** for global reasoning  
- **Global Spatial Attention (GSA)** for refined spatial boundaries  

Together, these mechanisms enable the model to understand **both global tumor context and fine structural boundaries**, improving overall segmentation quality.

---

## 🧩 Dataset and Preprocessing

### 📊 Dataset
- **Source:** *Figshare Brain MRI Dataset (T1-weighted contrast-enhanced)*  
- **Institutions:** Nanfang & Tianjin Medical University Hospitals  
- **Classes:** Glioma, Meningioma, Pituitary Tumor  
- **Total:** 3,064 samples  
- **Split:** 80% Train · 10% Validation · 10% Test  

### 🧹 Preprocessing
1. Image resize → 256×256 pixels  
2. Normalization → range [0, 1]  
3. Grayscale conversion for masks  
4. Augmentation → random rotation, flip, contrast enhancement  
5. Conversion → tensorized input (PyTorch DataLoader)  

These steps reduce overfitting and stabilize training under limited dataset conditions.

---

## 🧱 Architecture Design

### 🧩 Structural Overview
The model follows an **encoder–decoder** paradigm with an integrated **Feature Pyramid Network (FPN)** and **Transformer bridge**.

```text
Input MRI → Encoder → FPN fusion → Transformer Bridge → GSA → Decoder → Output Mask
```

---

### 🔹 Encoder
The encoder employs five convolutional layers (3×3 kernels, stride 1, padding 1) followed by batch normalization and ReLU.  
Each layer progressively reduces spatial resolution while increasing semantic abstraction.  

---

### 🔹 Feature Pyramid Network (FPN)
U-Net’s direct skip-connections tend to cause semantic mismatches between encoder and decoder features.  
**FPN** solves this by introducing top-down lateral connections that merge deep and shallow features through learnable 1×1 convolutions:

![FPN Equation](https://latex.codecogs.com/png.image?\color{black}F_{p_i}=Conv_{1\times1}(F_{l_i})+Upsample(F_{p_{i+1}}))

This enables **multi-scale context reconstruction**, improving generalization for tumors of varying size and morphology.

---

### 🔹 Transformer Bridge
The core of ATRU-Net is its **self-attention Transformer encoder**. Flattened FPN features are processed as token embeddings:  

![Attention Equation](https://latex.codecogs.com/png.image?\color{black}Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V)

This mechanism captures **long-range feature dependencies** across the entire image, giving the network an understanding of **global tumor structure** rather than isolated regions.

---

### 🔹 Global Spatial Attention (GSA)
GSA improves spatial sensitivity by combining average and max pooling over the feature maps, followed by convolution and sigmoid activation:  

![GSA Equation](https://latex.codecogs.com/png.image?\color{black}M_{GSA}=\sigma(f^{7\times7}([AvgPool(F);MaxPool(F)])))

This weighting map emphasizes **boundary-relevant** and **region-contrasting** pixels, crucial for precise tumor contouring.

---

### 🔹 Decoder
The decoder up-samples attention-enhanced feature maps and reconstructs spatial resolution.  
Each step includes skip-connections from the corresponding FPN stage, followed by 3×3 convolution, batch norm, and ReLU.  
A final 1×1 convolution outputs the segmentation mask with sigmoid activation.

---

## ⚙️ Training Configuration

| Setting | Value |
|:--|:--|
| Framework | PyTorch 2.0 |
| GPU | NVIDIA Tesla P100 |
| Epochs | 20 |
| Batch Size | 8 |
| Optimizers | Adam, AdamW, RMSProp, AdaGrad |
| Loss Function | Dice Loss |
| Activation | ReLU, GELU |
| Learning Rate | 0.001, 0.0001 |
| Metrics | Dice Coefficient, IoU |

---

## 🧠 Evaluation Results

| Model | Dice | IoU | Δ Dice | Δ IoU |
|:--|:--:|:--:|:--:|:--:|
| U-Net (baseline) | 0.7777 | 0.6477 | — | — |
| ATRU-Net (Adam) | 0.8476 | 0.7463 | +8.99% | +15.2% |
| ATRU-Net (AdamW) | 0.8616 | 0.7660 | +10.8% | +18.3% |
| **ATRU-Net (AdaGrad)** | **0.8827** | **0.8292** | **+13.5%** | **+28.0%** |


---

## 💬 Key Insights
- The **Transformer** module captures global tumor context overlooked by CNN filters.  
- **FPN** enables consistent feature propagation across scales.  
- **GSA** enhances tumor boundary clarity, reducing edge blur.  
- The **AdaGrad optimizer** produced smoother convergence due to adaptive step scaling.  

The hybridization of convolutional and transformer elements yields a model that is **accurate**, **interpretable**, and **efficient** for clinical use.

---

## ⚗️ Performance Analysis
- **Computation Time:** 34 minutes per training run (20 epochs).  
- **Precision:** High confidence localization on small lesions.  

---

## 🌍 Broader Impact
**Clinical relevance:** ATRU-Net enables more consistent tumor segmentation, aiding oncologists in early-stage detection and treatment planning.  
**Research significance:** Serves as a foundation for future medical image segmentation models integrating attention and transformer mechanisms.  
**Scalability:** Architecture can be extended to multi-modal inputs (CT, PET, Ultrasound).

---


## 📜 Conclusion
ATRU-Net demonstrates that combining **FPN**, **Transformer**, and **Global Spatial Attention** modules results in a **powerful hybrid architecture** that outperforms classical CNN-based models.  
With a **Dice score of 0.8827** and **IoU of 0.8292**, it marks a **13.5% improvement over U-Net**, providing new insights into **attention-guided medical image segmentation**.  

The model’s success at GEMASTIK XVII 2024 showcases its **technical innovation** and **research depth**, validated through reproducible experimentation and rigorous evaluation.

---

## 🏆 Competition Recognition

> 🥉 **3rd Place Winner — GEMASTIK XVII (2024)**  
> *Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi (ICT Scientific Paper Division)*  
> Officially organized by **KEMENRISTEKDIKTI (The Ministry of Education, Culture, Research, and Technology, Republic of Indonesia)**  
> Selected among **386 national teams** for exceptional research and model innovation.

---

## 👥 Authors and Roles

| Name | Role | Contribution |
|:--|:--|:--|
| **Krisna Bayu Dharma Putra** | **Team Leader · Artificial Intelligence Engineer · Model Architect** | Designed and coded the entire ATRU-Net architecture (FPN, GSA, Transformer), implemented training pipeline, ran ablation and evaluation studies, fine-tuning the model, and authored the technical and results sections of the paper |
| **Rabbani Nur Kumoro** | **Quality Assurance Engineer · Fine-tuning Specialist** | Performed fine-tuning and validation experiments post-implementation, ensured reproducibility and metric stability, and contributed to experiment logging |
| **Vincent Yeozekiel** | **Scientific Paper Writer · Visualization Specialist** | Created ATRU-Net architectural diagram, supported fine-tuning, formatted the research paper, and handled visualization of figures and result tables |
| **Dzikri Rahadian Fudholi** | **Supervisor** | Supervising the research |

📧 [linkedin.com/in/dharma-putra1305](https://linkedin.com/in/dharma-putra1305)  
🌐 [github.com/kbdp1305](https://github.com/kbdp1305)
