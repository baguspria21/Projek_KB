import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Fungsi untuk mendapatkan tanda tangan dan label dari semua subfolder
def get_signatures_and_labels(main_path="E:\Final project\dataset"):
    signatures = []
    labels = []
    label_names = {}
    current_label = 0

    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Membaca gambar
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  # Lewati jika gambar tidak valid

                # Preprocessing gambar tanda tangan
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                resized = cv2.resize(binary, (150, 150))

                # Simpan tanda tangan dan labelnya
                signatures.append(resized)
                labels.append(current_label)

            current_label += 1

    return signatures, labels, label_names

# Ambil tanda tangan dan label
signatures, labels, label_names = get_signatures_and_labels()

# Membuat dan melatih model SVM
if len(signatures) > 0:
    # Mengubah tanda tangan menjadi array datar (1D) untuk pelatihan SVM
    signatures_flattened = [signature.flatten() for signature in signatures]

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Pisahkan dataset menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(signatures_flattened, labels_encoded, test_size=0.2, random_state=42)

    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = clf.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi model: {accuracy * 100:.2f}%")

    # Simpan model pelatihan dan encoder
    joblib.dump(clf, 'svm_signature_model.pkl')
    joblib.dump(le, 'label_encoder_signature.pkl')
else:
    print("Dataset kosong atau tidak valid. Pastikan dataset berisi gambar tanda tangan.")
    exit()

# Load model pelatihan yang telah disimpan
clf = joblib.load('svm_signature_model.pkl')
le = joblib.load('label_encoder_signature.pkl')

# Fungsi untuk mengenali tanda tangan dalam gambar
def upload_and_recognize_signature(image_path):
    # Membaca gambar
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Gambar tidak valid. Pastikan path benar.")
        return

    # Preprocessing gambar tanda tangan
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(binary, (150, 150))
    signature_flattened = resized.flatten().reshape(1, -1)

    # Prediksi identitas tanda tangan
    label_encoded = clf.predict(signature_flattened)

    # Mendapatkan probabilitas dari model SVM
    proba = clf.predict_proba(signature_flattened)
    confidence = np.max(proba)  # Ambil probabilitas tertinggi sebagai confidence

    # Decode label
    label = le.inverse_transform(label_encoded)[0]
    name = label_names.get(label, "Unknown")

    print(f"Tanda tangan terdeteksi: {name} dengan confidence {int(confidence * 100)}%")

# Uji deteksi pada gambar
image_path = input("Masukkan path gambar tanda tangan: ")
upload_and_recognize_signature(image_path)
