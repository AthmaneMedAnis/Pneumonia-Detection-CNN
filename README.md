# Pneumonia-Detection-CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

The model is a Deep Learning model designed to assist medical professionals in detecting pneumonia from chest X-ray images. 

In medical diagnosis, the cost of missing a positive case (False Negative) is extremely high. Therefore, this project prioritizes **Recall (Sensitivity)** over pure Accuracy. The goal is to ensure that virtually every case of pneumonia is detected, minimizing the risk of discharging a sick patient.

This project utilizes **Transfer Learning** with a VGG16 architecture pre-trained on ImageNet.

## 📊 Key Results

The model was evaluated on an unseen Test Set of 624 images.

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **98.46%** | 🥇 **Primary Goal.** The model detects ~98% of pneumonia cases. |
| **Accuracy** | **91.02%** | The model makes a correct prediction 9 times out of 10. |
| **Precision** | **88.48%** | The False Positive rate is acceptable for a screening tool. |

> *"The high recall score demonstrates the model's reliability as a preliminary screening tool for radiologists."*

## 🧠 Methodology

### 1. Data Preprocessing
* **Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle.
* **Augmentation:** To prevent overfitting on the small dataset, training images undergo random rotations (20°), zooms (20%), and horizontal flips.
* **Normalization:** Pixel values are rescaled to the [0, 1] range.

### 2. Model Architecture
We use a **Transfer Learning** approach:
* **Backbone:** **VGG16** (pre-trained on ImageNet) is used as a feature extractor. The convolutional base is frozen.
* **Head:** A custom classification head is added:
    * `GlobalAveragePooling2D`
    * `Dense` layer (ReLU activation)
    * `Dropout` (0.5) to reduce overfitting
    * `Dense` Output layer (Sigmoid activation)

### 3. Training Strategy
* **Optimizer:** Adam.
* **Loss Function:** Binary Crossentropy.
* **Callbacks:**
    * `EarlyStopping`: Monitors validation loss to prevent overfitting.
    * `ModelCheckpoint`: Saves the best model based on **Validation Recall**.

## 📂 Project Structure

```bash
├── data/               # Dataset (Train/Test/Val)
├── notebooks/          # Jupyter Notebooks for training
├── models/             # Saved .keras models
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/DeepBreath.git](https://github.com/your-username/DeepBreath.git)
    cd DeepBreath
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib pandas
    ```

3.  **Download the Data:**
    Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and extract it into the `data/` folder.

4.  **Run the Notebook:**
    Open `Pneumonia_CNN.ipynb` in Jupyter or Google Colab to train the model or evaluate the results.

## 🔮 Future Improvements

* **Explainability:** Implement **Grad-CAM** to visualize the specific regions of the lung the model focuses on.
* **Fine-Tuning:** Unfreeze the last block of VGG16 to potentially improve precision.
* **Deployment:** Wrap the model in a FastAPI container for real-time inference.

## 🤝 Acknowledgements

* Dataset provided by [Paul Mooney](https://www.kaggle.com/paultimothymooney) on Kaggle.
* Original VGG16 paper: Simonyan & Zisserman (2014).
