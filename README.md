# Emotion Detection Project Report

Team Name :
Member 1, SRN: Sairam S, PES1UG23AM257, [@s-sairam](https://github.com/s-sairam)

Member 2, SRN: Shama Hegde, PES1UG23AM278, [@shamaphegde](https://github.com/shamaphegde)

Member 3, SRN: Tanmaya Karanth, PES1UG23AM335, [@tanmayakaranth](https://github.com/tanmayakaranth)





## 1. Approach

### 1.1 Problem Formulation
We treat emotion detection as a supervised multi-class classification problem. 

* **Input:** A sentence
* **Output:** One emotion label
* **Goal:** Learn a mapping function
```math
 f(\text{sentence}) \to \text{emotion\_class}
```

We fine-tune a pretrained language model instead of training from scratch to leverage **transfer learning**.

### 1.2 Dataset
* **Source:** Hugging Face Dataset Hub
* **Dataset Name:** `shreyaspullehf/emotion_dataset_100k`
* **Data Splitting:**
    * **80%** Training
    * **20%** Testing
* **Validation:** Training data further split into **90% Training** and **10% Validation** (stratified by emotion class).

**Exploratory Data Analysis (EDA):**
We visualized the class distribution via bar plots and sentence length via histograms. 
> **Observation:** Most sentences are short ($< 40$ words), leading us to set a `max_length = 64`.

### 1.3 Label Encoding
Emotion labels were converted into integer IDs using `label2id` and `id2label` mappings. This is a prerequisite for `CrossEntropyLoss`, which expects class indices.

### 1.4 Tokenization
* **Model:** `distilbert-base-uncased`
* **Tokenizer:** `DistilBertTokenizer`
* **Settings:** * Padding: `True`
    * Truncation: `True`
    * Max Length: 64
    * Output: PyTorch tensors

### 1.5 Model Architecture
We implemented a custom classifier on top of the transformer backbone:



1.  **DistilBERT** (Pretrained backbone)
2.  **CLS Token Representation** (Extracting `last_hidden_state[:,0,:]`)
3.  **Dropout (0.3)** (Regularization layer)
4.  **Linear Layer** (Mapping to `num_classes`)
5.  **Logits** (Final output)

---

### 1.6 Training Setup

| Component | Value |
| :--- | :--- |
| **Loss Function** | `CrossEntropyLoss` |
| **Optimizer** | `AdamW` |
| **Learning Rate** | $1 \times 10^{-3}$ |
| **Batch Size** | 32 |
| **Epochs** | 3 |
| **Gradient Clipping** | 1.0 |

**Training Steps:**
1. Forward pass
2. Compute loss
3. Backpropagation
4. Gradient clipping
5. Optimizer step
6. Validation after each epoch

---

### 1.7 Evaluation Metrics
On the test set, we computed:
* **Accuracy**
* **Precision** (Weighted)
* **Recall** (Weighted)
* **F1-score** (Weighted)
* **Confusion Matrix**

*Note: Weighted averaging was utilized to account for class imbalance.*

### 1.8 Inference Pipeline
The `predict_text()` function handles the end-to-end flow:
* Tokenizes input text.
* Performs forward pass.
* Applies **Softmax** to convert logits to probabilities.
* Returns predicted emotion and confidence score.

---

## 2. Assumptions

### Dataset Assumptions
* Each sentence belongs to exactly one emotion (**single-label**).
* Labels are clean and accurately represent the text.
* Class imbalance is moderate enough for weighted metrics to handle.

### Modeling Assumptions
* The **[CLS] token** sufficiently captures the semantic meaning of the sentence.
* `max_length = 64` captures enough context for short-form emotional text.
* 3 epochs are sufficient for the model to converge without overfitting.

### Optimization Assumptions
* A learning rate of $1 \times 10^{-3}$ is appropriate for this specific head/backbone combo.
* No scheduler is needed for this short training duration.

---

## 3. Observations

### Data Insights
* Emotional language often relies on **strong adjectives**.
* Some emotions (e.g., Joy, Sadness) appear more frequently in the dataset than others.

### Training Behavior
* **Good Generalization:** Training loss and validation loss decreased in tandem.
* **Overfitting Risk:** If training loss continues to drop while validation loss rises, the model is memorizing the data.

### Model Performance
The model achieved high accuracy, but we noted semantic confusion in these pairs:
* *Sadness* vs. *Loneliness*
* *Anger* vs. *Frustration*
* *Fear* vs. *Anxiety*

---

## 4. Limitations
* No extensive **hyperparameter tuning**.
* Lack of a **Learning Rate Scheduler**.
* No **Early Stopping** mechanism.
* No comparison between different architectures (e.g., RoBERTa or XLNet).

---

## 5. Future Improvements
1.  **Lower Learning Rate:** Use $2 \times 10^{-5}$ (industry standard for BERT fine-tuning).
2.  **Schedulers:** Implement a linear warmup with decay.
3.  **Loss Weighting:** Use class weights in `CrossEntropyLoss` to handle imbalance.
4.  **Layer Freezing:** Freeze the DistilBERT base for the first epoch to stabilize the head.
5.  **K-Fold Cross-Validation:** Ensure robustness across different data splits.

---

## 📌 Conclusion
This project demonstrates a robust pipeline for emotion classification using **Transformer-based Transfer Learning**. By leveraging DistilBERT, we achieved a balance between computational efficiency and high predictive performance.
