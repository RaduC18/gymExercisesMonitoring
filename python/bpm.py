import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from scipy.signal import butter, filtfilt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


af_folder = "./mimic_perform_af_csv"
non_af_folder = "./mimic_perform_non_af_csv"
output_npz = "ppg_af_dataset.npz"
output_model = "ppg_cnn_lstm_model.h5"
win_size = 185


def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=125.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def load_ppg_data(folder_path, label, max_files=None):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    data = []
    labels = []

    if max_files:
        all_files = all_files[:max_files]

    for file in all_files:
        df = pd.read_csv(file)
        df.columns = [col.strip().lower() for col in df.columns]

        if "ppg" in df.columns:
            ppg = df["ppg"].dropna().values
            ppg = butter_bandpass_filter(ppg)

            for i in range(0, len(ppg) - win_size, 250):
                segment = ppg[i:i + win_size]
                if len(segment) == win_size:
                    noise = np.random.normal(0, 0.005, segment.shape)
                    data.append(segment + noise)
                    labels.append(label)

    return np.array(data), np.array(labels)


af_data, af_labels = load_ppg_data(af_folder, label=1)
non_af_data, non_af_labels = load_ppg_data(non_af_folder, label=0)


X = np.concatenate([af_data, non_af_data], axis=0)
y = np.concatenate([af_labels, non_af_labels], axis=0)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


max_val = np.max(np.abs(X_train))
X_train = X_train / max_val
X_val = X_val / max_val
X_test = X_test / max_val


X_train = X_train.reshape((-1, win_size, 1))
X_val = X_val.reshape((-1, win_size, 1))
X_test = X_test.reshape((-1, win_size, 1))


num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y_train)
cw_dict = dict(enumerate(cw))


model = Sequential()
model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', input_shape=(win_size, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
]


history = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val_cat),
    class_weight=cw_dict,
    callbacks=callbacks
)


test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n Test accuracy: {test_acc:.4f}\n")


y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_cat, axis=1)

print(confusion_matrix(y_true_labels, y_pred_labels))
print(classification_report(y_true_labels, y_pred_labels))


model.save(output_model)

np.savez(output_npz,
         X_train=X_train, y_train=y_train_cat,
         X_val=X_val, y_val=y_val_cat,
         X_test=X_test, y_test=y_test_cat,
         input_shape=np.array([win_size, 1]),
         class_weights=cw_dict)


plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("CNN + LSTM Model Accuracy - PPG")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()