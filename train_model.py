import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "dataset/train"  # Must contain 'rotten' (class 0) and 'fresh' (class 1) subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["rotten", "fresh"],  # 0 = rotten, 1 = fresh
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["rotten", "fresh"],
    subset="validation"
)

# 2. Build MobileNetV2 Transfer Learning Architecture
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers for fast training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
# Sigmoid outputs a continuous probability between 0.0 (rotten) and 1.0 (fresh)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 3. Train and Save Model
model.fit(train_gen, validation_data=val_gen, epochs=5)
model.save("freshness_model.h5")
print("Saved freshness_model.h5 successfully!")