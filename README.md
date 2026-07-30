# Pneumonia-Detection-CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-red)

## Project Overview

The model is a Deep Learning model designed to assist medical professionals in detecting pneumonia from chest X-ray images.

In medical diagnosis, the cost of missing a positive case (False Negative) is extremely high. Therefore, this project prioritizes **Recall (Sensitivity)** over pure Accuracy. The goal is to ensure that virtually every case of pneumonia is detected, minimizing the risk of discharging a sick patient.

This project utilizes **Transfer Learning** with a VGG16 architecture pre-trained on ImageNet.

The model has been optimized using TensorFlow Lite and deployed as a Serverless REST API on AWS Lambda, demonstrating an end-to-end pipeline from training to cloud inference.

## Key Results

The model was evaluated on an unseen Test Set of 624 images.

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **98.46%** | **Primary Goal.** The model detects ~98% of pneumonia cases. |
| **Accuracy** | **91.02%** | The model makes a correct prediction 9 times out of 10. |
| **Precision** | **88.48%** | The False Positive rate is acceptable for a screening tool. |

> *"The high recall score demonstrates the model's reliability as a preliminary screening tool for radiologists."*

## Methodology

### 1. Data Preprocessing

* **Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle.
* **Augmentation:** To prevent overfitting on the small dataset, training images undergo random rotations (20°), zooms (20%), and horizontal flips.
* **Normalization:** Pixel values are rescaled to the [0, 1] range.

### 2. Model Architecture

We use a **Transfer Learning** approach:

* **Backbone:** **VGG16** (pre-trained on ImageNet) is used as a feature extractor. The convolutional base is frozen.
* **Head:** A custom classification head is added:
  * `Flatten` layer to convert 2D feature maps into a 1D vector.
  * `Dense` Output layer (1 neuron) with **Sigmoid** activation for binary classification.

### 3. Training Strategy

* **Optimizer:** Adam.
* **Loss Function:** Binary Crossentropy.
* **Callbacks:**
  * `EarlyStopping`: Monitors **Validation Recall** (stops if sensitivity doesn't improve for 3 epochs).
  * `ModelCheckpoint`: Saves the best model based on **Validation Recall**.

## Cloud Deployment & MLOps

To ensure scalability and cost-effectiveness, the inference engine is deployed using a serverless architecture.

* **Model Optimization:** The trained VGG16 model was converted to TensorFlow Lite (`.tflite`), significantly reducing its footprint while maintaining diagnostic accuracy.
* **Serverless Inference:** The application is hosted on **AWS Lambda** (Python 3.13 runtime) using `ai-edge-litert` to keep the deployment package lightweight.
* **Storage & Updates:** Large deployment packages were routed through **Amazon S3** to bypass standard Lambda size limits.

## Project Structure

```bash
├── DATA/                      # Dataset (Train/Test/Val)
├── models/
│   └── pneumonia_model.tflite # Quantized model for production
├── lambda_function.py         # AWS Lambda handler script  
├── lambda_test.py             # Local script to test the live API
├── Pneumonia_CNN.ipynb        # Jupyter Notebook for training
├── .env.example           # Template for environment variables
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## How to Run

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AthmaneMedAnis/Pneumonia-Detection-CNN.git
    cd Pneumonia-Detection-CNN
    ```

2. **Install dependencies:**

    ```bash
    pip install tensorflow
    ```

3. **Download the Data:**
    Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and extract it into the `data/` folder.

4. **Run the Notebook:**
    Open `Pneumonia_CNN.ipynb` in Jupyter or Google Colab to train the model or evaluate the results.

5. **Test the Cloud API (Live Inference):**
    To test the live AWS Lambda deployment without retraining the model:
    * Create a .env file in the root directory and add the AWS Lambda URL.
    * Run the test script with a sample image:

     ```bash
    python lambda_test.py
    ```

## Future Improvements

* **Explainability:** Implement **Grad-CAM** to visualize the specific regions of the lung the model focuses on.
* **Fine-Tuning:** Unfreeze the last block of VGG16 to potentially improve precision.
* **Front-End Interface**: Develop a lightweight web application using Streamlit to allow users to drag-and-drop X-rays for instant clinical feedback.
* 

## Acknowledgements

* Dataset provided by [Paul Mooney](https://www.kaggle.com/paultimothymooney) on Kaggle.
* Original VGG16 paper: Simonyan & Zisserman (2014).
