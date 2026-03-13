# Task 12 Model Report

## Dataset
- MNIST loaded from Hugging Face: `ylecun/mnist`.
- Images flattened to 784 features and normalized to [0, 1].

## Model and Training
- Model: Linear(784->256) + ReLU + Linear(256->128) + ReLU + Linear(128->10).
- Epochs: 5
- Batch size: 128
- Learning rate: 0.001
- Optimizer: AdamW
- Loss: CrossEntropyLoss

## Final Results
- Final validation loss: 0.094840
- Final validation accuracy: 0.9718
- Test loss: 0.080691
- Test accuracy: 0.9748
- Curves saved at `data-science-12-training-curves.png`.