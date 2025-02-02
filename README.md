# Federated Learning for ECG Classification

## Overview
This project implements a federated learning approach for binary classification of ECG signals from the MIT-BIH dataset. The goal is to simulate a decentralized healthcare scenario where multiple hospitals contribute to training a shared model without sharing patient data.

## Dataset
The dataset can be downloaded from the following link:
[MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/1.0.0/)
The project uses the **MIT-BIH Arrhythmia Dataset**, preprocessed into train and test CSV files.

- `mitbih_train.csv` - Training data
- `mitbih_test.csv` - Test data

## Project Structure
- **Data Preprocessing**: Loads and normalizes ECG data, converting it into tensors.
- **Federated Learning Simulation**: Splits training data among three simulated hospitals.
- **CNN Model**: A 1D Convolutional Neural Network (CNN) for ECG classification.
- **Federated Training**: Each hospital trains locally, and the model is aggregated centrally.
- **Evaluation**: The trained model is evaluated on a test set.
- **Model Saving**: The final global model is saved as `federated_global_model.pth`.

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install torch pandas
```

## Running the Code
Execute the script to train the model using federated learning:
```bash
python federated_ecg.py
```

## Model Architecture
The CNN model consists of:
- **Conv1D layers** for feature extraction
- **MaxPooling layers** for dimensionality reduction
- **Fully connected layers** for classification

## Training Process
1. **Data Loading**: Splits training data across three hospitals.
2. **Training Loop**: Each hospital trains its local model.
3. **Federated Learning**: Models are aggregated after training rounds.
4. **Evaluation**: The final model is tested on unseen data.

## Results
After training, the accuracy on the test set is displayed. The trained model can be used for further analysis or real-world deployment.

## License
This project is open-source and available for research and educational purposes.

## Acknowledgments
- MIT-BIH Arrhythmia Database
- PyTorch for deep learning implementation


