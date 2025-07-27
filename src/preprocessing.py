def preprocess_image(image_path, img_size=(64,64)): # Grayscaling, resizing, normalizing & flatting the images.
    img = Image.open(image_path).convert("L") # Grayscale
    img = img.resize(img_size) # Resizing
    img_array = np.array(img).flatten().astype(np.float32) / 255.0 # Flatting and  Normalizing
    
    return img_array
