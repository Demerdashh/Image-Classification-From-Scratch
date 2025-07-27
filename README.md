# 🧠 Logistic Regression from Scratch — Binary Image Classifier

This project is a full implementation of **logistic regression from scratch**, using **only NumPy** — no machine learning libraries or frameworks. It classifies images of **cats vs dogs** by manually constructing the entire training pipeline, from data preprocessing to gradient descent.

> **Inspired by** [deeplearning.ai’s Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) by Andrew Ng.

---

## 🚀 Features

- ✅ **Pure NumPy** — No ML libraries used (e.g. scikit-learn, TensorFlow, Keras)
- ✅ Manual implementation of:
  - Sigmoid activation
  - Forward and backward propagation
  - Cost computation
  - Parameter optimization with gradient descent
  - Prediction and evaluation
- ✅ Custom `train/test` split logic (no `train_test_split`)
- ✅ Trains on real images from [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- ✅ Cost visualization over iterations

---

## 🗂️ Folder Structure
<img width="592" height="279" alt="image" src="https://github.com/user-attachments/assets/eb4c0787-e0bc-466f-b333-48e7bb02bc9a" />




---

## 🧪 Dataset

- **Dataset**: [Dogs vs Cats – Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- **Sample size**: 2000 cat images + 2000 dog images
- **Preprocessing**:
  - Grayscale conversion
  - Resizing to 64×64
  - Flattened into 1D vectors
  - Normalized pixel values to `[0, 1]`

📁 _See `data/README.md` for dataset link and folder setup instructions._

---

## 📜 Methodology

The project strictly avoids using any prebuilt ML training tools. Everything is implemented manually, including:

### 🔄 Preprocessing
- Grayscale conversion, resizing, normalization, flattening
- Custom `train/test` splitting logic

### 🧮 Model
- Manual parameter initialization
- Sigmoid-based hypothesis
- Binary cross-entropy loss
- Vectorized forward & backward propagation
- Gradient descent optimization

### 📈 Evaluation
- Manual prediction and accuracy calculation
- Optional cost plot to track learning

---

## ❌ No Machine Learning Libraries Used

| Library        | Used? |
|----------------|-------|
| NumPy          | ✅     |
| scikit-learn   | ❌     |
| TensorFlow     | ❌     |
| PyTorch        | ❌     |
| OpenCV         | ❌     |

All logic is hand-coded to reinforce mathematical understanding.

---

## 💡 Inspiration

This project is heavily inspired by the foundational ideas taught in:

> 🎓 **deeplearning.ai – Neural Networks and Deep Learning**  
> by [Andrew Ng](https://www.andrewng.org/)

Additional self-imposed challenges:
- Implemented custom `train_test_split`
- Used a row-wise matrix design (images as row vectors instead of column vectors)
- Trained on a real-world image dataset
- Visualized training cost for inspection

---

## 🧑‍💻 Author

**Youssef Eldemerdash**  
_Passionate about learning by building things from first principles._

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.


