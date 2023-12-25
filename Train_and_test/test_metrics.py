import pickle
import numpy as np
import json

from keras.utils import to_categorical
from keras.models import load_model

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

right_hand_model = load_model("../Train_and_test/right_hand_model.h5")
right_hand_landmarks = pickle.load(open("../Data_preprocessing/right_hand_test_data.pickle", "rb"))

right_hand_data = right_hand_landmarks['data']
right_hand_labels = right_hand_landmarks['labels']

def cut_landmarks(data, labels):
    new_data = []
    new_labels = []

    for landmarks, label in zip(data, labels):
        if len(landmarks) == 126:
            new_landmarks = landmarks[:63] 
            new_data.append(new_landmarks)
            new_labels.append(label)
        else:
            new_data.append(landmarks)
            new_labels.append(label)

    data = new_data
    labels = new_labels

    return data, labels

def preprocessing_training(data, labels):
    data_np = np.array(data)
    labels_np = np.array(labels)
   
    train_sample = len(data_np)
   
    timesteps = 1  
    features = len(data_np[0])

    data = data_np.reshape(train_sample, timesteps, features)
    labels = to_categorical(labels_np)
    
    return data, labels

right_hand_data, right_hand_labels = cut_landmarks(right_hand_data, right_hand_labels)
right_hand_data, right_hand_labels = preprocessing_training(right_hand_data, right_hand_labels)

res = right_hand_model.predict(right_hand_data)
predicted_labels = np.argmax(res, axis=1)  
true_labels = np.argmax(right_hand_labels, axis=1)  

accuracy = np.mean(predicted_labels == true_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

report = classification_report(true_labels, predicted_labels)
print(report)


unique_labels = np.unique(true_labels)  
accuracy_per_label = {} 

for label in unique_labels:
    indices = np.where(true_labels == label)  
    accuracy = accuracy_score(true_labels[indices], predicted_labels[indices])  
    accuracy_per_label[label] = accuracy  

file_path = "../Train_and_test/Index_to_letters.json"
with open(file_path, 'r') as json_file:
    data = json.load(json_file)
reversed_data = {v: k for k, v in data.items()}

labels_dict =  [reversed_data[i] for i in range(len(reversed_data))]

print("Accuracy per label:")
for label, accuracy in accuracy_per_label.items():
    print(f"Label {labels_dict[label]}: Accuracy {accuracy * 100:.2f}%")

labels = list(accuracy_per_label.keys())
accuracy_values = list(accuracy_per_label.values())