import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
data = pd.read_csv(r'dataset\clean\DR_dataset.csv')

class Train_FCNN:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def preprocess_data(self):
        cols = list(self.data.columns)
        cols.insert(0, cols.pop(cols.index("label")))
        self.data = self.data[cols]

        features = self.data.drop(columns=['label']).values
        labels = self.data['label']

        labels = self.label_encoder.fit_transform(labels)
        labels = to_categorical(labels, num_classes=5)

        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        self.model = Sequential([
            Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002), input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(5, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1  # Shows full training details
        )

        # Get final accuracy & loss
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print("\n Training Completed!")
        print(f" Final Training Accuracy: {final_train_acc * 100:.2f}%")
        print(f" Final Validation Accuracy: {final_val_acc * 100:.2f}%")
        print(f" Final Training Loss: {final_train_loss:.4f}")
        print(f" Final Validation Loss: {final_val_loss:.4f}")
        return history
    
    def save_model(self, filepath='model.h5'):
        self.model.save(filepath)
        print("Model saved successfully.")
    
    def load_model(self, filepath='model.h5'):
        self.model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully.")

# Example usage
trainer = Train_FCNN(data)
X_train, X_test, y_train, y_test = trainer.preprocess_data()
trainer.build_model(input_shape=X_train.shape[1])
trainer.train_model(X_train, y_train, X_test, y_test)
