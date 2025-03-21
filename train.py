import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import pickle
from services.feature_extraction.EfficientNet import EfficientNetExtractor

class Train_FCNN:
    def __init__(self, data, pca_components=300):
        self.data = data
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=pca_components)
    
    def preprocess_data(self):
        print("Unique labels in dataset:", self.data['label'].unique())
        
        features = self.data.drop(columns=['label']).values
        labels = self.data['label']
        
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, encoded_labels)
        
        unique, counts = np.unique(labels_resampled, return_counts=True)
        print("Class distribution after SMOTE:", dict(zip(unique, counts)))
        
        labels_categorical = to_categorical(labels_resampled, num_classes=len(self.label_encoder.classes_))
        features_resampled = self.scaler.fit_transform(features_resampled)
        
        # Apply PCA
        features_resampled = self.pca.fit_transform(features_resampled)
        
        X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_categorical, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    # def build_model(self, input_shape):
    #     self.model = Sequential([
    #         Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002), input_shape=(input_shape,)),
    #         BatchNormalization(),
    #         Dropout(0.5),
            
    #         Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    #         BatchNormalization(),
    #         Dropout(0.4),
            
    #         Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    #         BatchNormalization(),
    #         Dropout(0.3),
            
    #         Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
    #         BatchNormalization(),
    #         Dropout(0.3),
            
    #         Dense(len(self.label_encoder.classes_), activation='softmax')
    #     ])

    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    #     self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    def build_model(self, input_shape):
        self.model = Sequential([
            Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0005), input_shape=(input_shape,)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),

            Dense(256),
            # BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),

            Dense(128),
            # BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),

            Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),

            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Reduce learning rate slightly
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )

        print(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
        return history
    
    def save_model(self, filepath='model.h5'):
        self.model.save(filepath)
        print("Model saved successfully.")
    
    def save_label_encoder(self, filepath='label_encoder.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {filepath}")
    
    def load_model(self, image, model_filepath='model.h5', encoder_filepath='label_encoder.pkl'):
        self.model = tf.keras.models.load_model(model_filepath)
        with open(encoder_filepath, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        efficientnet_extractor = EfficientNetExtractor().load_model()
        extracted_features = efficientnet_extractor.extract_features(img_path=image).flatten()
        
        features = self.scaler.transform([extracted_features])
        
        # Apply PCA before testing
        features = self.pca.transform(features)
        
        prediction = self.model.predict(features)
        predicted_class = np.argmax(prediction)
        predicted_class_name = self.label_encoder.inverse_transform([predicted_class])[0]
        print(f"Predicted class: {predicted_class_name}")
        
        return predicted_class_name

# Example usage
if __name__ == "__main__":
    data = pd.read_csv(r'dataset\clean\DR_dataset.csv')
    trainer = Train_FCNN(data, pca_components=1000)  # Adjust components as needed
    X_train, X_test, y_train, y_test = trainer.preprocess_data()
    trainer.build_model(input_shape=X_train.shape[1])
    trainer.train_model(X_train, y_train, X_test, y_test)
    trainer.save_model("model.h5")
    trainer.save_label_encoder("label_encoder.pkl")
    
    image = r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\raw\Diabetic retinopathy\Moderate\fd48cf452e9d.png"
    image = r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\raw\Diabetic retinopathy\No_DR\ffc04fed30e6.png"
    image = r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\raw\Diabetic retinopathy\Severe\1ab3f1c71a5f.png"
    trainer.load_model(image, "model.h5", "label_encoder.pkl")
