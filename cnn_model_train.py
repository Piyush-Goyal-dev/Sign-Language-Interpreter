import numpy as np
import pickle
import cv2, os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_size():
    try:
        img = cv2.imread('gestures/0/100.jpg', 0)
        if img is None:
            all_images = glob('gestures/*/*.jpg')
            if len(all_images) > 0:
                img = cv2.imread(all_images[0], 0)
        return img.shape
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def get_num_of_classes():
    num_classes = len(glob('gestures/*'))
    if num_classes == 0:
        print("Error: No gesture folders found!")
        exit(1)
    return num_classes

print("=" * 60)
print("IMPROVED CNN MODEL TRAINING")
print("=" * 60)

image_x, image_y = get_image_size()
print(f"\n✓ Image size: {image_x} x {image_y}")

num_classes = get_num_of_classes()
print(f"✓ Number of gesture classes: {num_classes}")

if num_classes < 3:
    print("\n⚠ WARNING: You only have", num_classes, "gesture(s)!")
    print("For better accuracy, create at least 5-10 gestures.")
    print("Different gestures should look VERY different from each other.")
    cont = input("\nContinue anyway? (y/n): ")
    if cont.lower() != 'y':
        exit(0)

def improved_cnn_model():
    """Improved CNN architecture with better accuracy"""
    num_of_classes = get_num_of_classes()
    
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(32, (3,3), input_shape=(image_x, image_y, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Conv Block
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third Conv Block
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    # Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    
    # Use Adam optimizer (better than SGD)
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'cnn_model_keras2.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=0.00001
    )
    
    callbacks_list = [checkpoint, early_stop, reduce_lr]
    
    return model, callbacks_list

def train():
    print("\n" + "=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    
    try:
        with open("train_images", "rb") as f:
            train_images = np.array(pickle.load(f))
        print(f"✓ Loaded {len(train_images)} training images")
        
        with open("train_labels", "rb") as f:
            train_labels = np.array(pickle.load(f), dtype=np.int32)
        print(f"✓ Loaded {len(train_labels)} training labels")

        with open("val_images", "rb") as f:
            val_images = np.array(pickle.load(f))
        print(f"✓ Loaded {len(val_images)} validation images")
        
        with open("val_labels", "rb") as f:
            val_labels = np.array(pickle.load(f), dtype=np.int32)
        print(f"✓ Loaded {len(val_labels)} validation labels")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run 'python load_images.py' first!")
        exit(1)

    # Reshape and normalize
    print("\nPreprocessing data...")
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    # Normalize to 0-1
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    
    # One-hot encode labels
    num_classes = get_num_of_classes()
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    
    print(f"✓ Training data shape: {train_images.shape}")
    print(f"✓ Validation data shape: {val_images.shape}")

    # Data augmentation for better generalization
    print("\nSetting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(train_images)

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING IMPROVED MODEL")
    print("=" * 60)
    model, callbacks_list = improved_cnn_model()
    model.summary()
    
    # Train model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("Training with data augmentation...")
    print("This may take 15-45 minutes depending on your hardware...")
    print("Epochs: 30 (with early stopping), Batch size: 32\n")
    
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=(val_images, val_labels),
        epochs=30,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\n✓ Final Validation Accuracy: {scores[1]*100:.2f}%")
    print(f"✓ CNN Error: {100-scores[1]*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✓ Model saved as: cnn_model_keras2.h5")
    print("✓ You can now run: python final.py")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()