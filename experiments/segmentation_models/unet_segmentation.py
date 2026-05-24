# unet_tensorflow_experiment.py
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import cv2
import numpy as np

DATASET_PATH = "dataset"
IMG_HEIGHT, IMG_WIDTH = 128, 128

def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=1):
    inputs = Input(input_size)
    
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    u1 = UpSampling2D((2, 2))(c4)
    u1 = concatenate([u1, c3], axis=-1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
    u2 = UpSampling2D((2, 2))(c5)
    u2 = concatenate([u2, c2], axis=-1)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
    u3 = UpSampling2D((2, 2))(c6)
    u3 = concatenate([u3, c1], axis=-1)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs, outputs)
    return model

def load_data(img_dir, mask_dir):
    img_paths = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
    images, masks = [], []
    
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        images.append(img / 255.0)
        
        mask_path = Path(mask_dir) / f"mask_{img_path.name}"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), 0)
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
            masks.append(mask / 255.0)
        else:
            masks.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)))
    
    return np.array(images), np.array(masks).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

model = unet_model(num_classes=1)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

X_train, y_train = load_data(f"{DATASET_PATH}/images", f"{DATASET_PATH}/masks")
X_val, y_val = load_data(f"{DATASET_PATH}/val_images", f"{DATASET_PATH}/val_masks")

data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_val, y_val),
    epochs=20,
    steps_per_epoch=len(X_train) // 8
)

model.save("unet_defect_model.h5")

test_img = cv2.imread(f"{DATASET_PATH}/test/sample.jpg")
test_img = cv2.resize(test_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
pred_mask = model.predict(test_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3))
pred_mask = (pred_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
cv2.imwrite("predicted_defect_mask.png", pred_mask)