def preprocess_image(image_path, img_size=(64,64)): # Grayscaling, resizing, normalizing & flatting the images.
    img = Image.open(image_path).convert("L") # Grayscale
    img = img.resize(img_size) # Resizing
    img_array = np.array(img).flatten().astype(np.float32) / 255.0 # Flatting and  Normalizing
    
    return img_array

#loading dataset and Train-Test split
def load_dataset(cat_folder,dog_folder,img_size=(64,64), test_size=0.2, seed=42):
    images = []
    labels = []

    #loading cats
    for file_name in os.listdir(cat_folder):
        path = os.path.join(cat_folder,file_name)
        images.append(preprocess_image(path, img_size))
        labels.append(0) # 0 --> label of a cat
    
    #loading dogs
    for filename in os.listdir(dog_folder):
        path = os.path.join(dog_folder,file_name)
        images.append(preprocess_image(path, img_size))
        labels.append(1) # 1 --> label of a dog
    X = np.array(images)
    y = np.array(labels).reshape(-1,1)

    # Train-Test split 
    def custom_train_test_split(X, y, test_size=0.2):
        np.random.seed(42)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        split = int(len(X) * (1 - test_size))
        X_train = X[indices[:split]]
        X_test = X[indices[split:]]
        y_train = y[indices[:split]]
        y_test = y[indices[split:]]

        return X_train, X_test, y_train, y_test
    
    return custom_train_test_split(X, y, test_size)
