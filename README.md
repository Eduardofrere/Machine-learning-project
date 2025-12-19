
This project focuses on classifying chest X-ray images into four categories:

Normal

Pneumonia

Cardiomegaly

Pleural Effusion

We used part of the NIH ChestX-ray14 dataset, which contains over 112,000 X-ray images labeled for 14 different diseases. Since the goal of the project was to keep things manageable, I filtered the dataset down to just the four classes listed above.

The project includes:

data preprocessing and label filtering

balancing the dataset

training a deep learning model

evaluating the model on a test set

📦 Dataset & Preprocessing

The NIH dataset includes many disease labels, and some images contain multiple findings.
To make the problem simpler and more focused, We only kept the four classes we needed.
If an image had more than one of these labels, we applied this priority rule:

Pneumonia → Cardiomegaly → Effusion → Normal

This ensures each image is assigned one single label, even if multiple appear.

After filtering, we balanced the dataset so each class had the same number of images.
This helps prevent the model from being biased toward the more common classes (for example, "Normal" has over 60k examples originally).

The script prepare_dataset.py performs all these steps and generates a clean CSV file (four_class_labels.csv) that is used for training.

Model

For the classifier, we used ResNet-18 with pretrained ImageNet weights (transfer learning).
Instead of training a network from scratch, transfer learning lets the model start from useful features, which usually improves performance on medical images.

The model was trained using:

CrossEntropyLoss

Adam optimizer

data augmentation (random crops, flips, rotations)

70 / 15 / 15 train-val-test split

Results

Here are the main results from training:

Final training accuracy: ~81%

Best validation accuracy: ~58.5%

Test accuracy: ~53.5%

These numbers show that the model does learn useful patterns (well above random guessing, which would be 25% for four classes).
There is still a noticeable gap between training and validation accuracy, which suggests some overfitting. This is expected with a medical dataset and a relatively small balanced subset.

Project Structure
CHESTXRAY_PROJECT/
│
├── prepare_dataset.py     # Filters dataset, applies label priority, balances classes
├── train_chestxray.py     # Model training and evaluation (ResNet-18)
├── data/                  # Dataset folder (ignored in GitHub)
│   └── four_class_labels.csv
├── best_resnet18_chestxray.pth   # Saved model weights
└── README.md


Note: The raw images are not included in the repository because the dataset is very large (45GB+). The .gitignore file prevents these from being uploaded.

▶️ How to Run
1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

2. Install the packages
pip install torch torchvision pandas scikit-learn matplotlib

3. Prepare the dataset

Place the extracted NIH dataset files inside the data/ folder, then run:

python prepare_dataset.py

4. Train the model
python train_chestxray.py


This will save the best-performing model to
best_resnet18_chestxray.pth.