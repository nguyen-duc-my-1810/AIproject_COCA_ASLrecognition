import pickle
import numpy as np
from collections import Counter
import random

import Augmentation

data_right_hand = pickle.load(open("../Data_preprocessing/new_data.pickle", 'rb'))
data_right_hand = data_right_hand
data_right_hand_for_test = data_right_hand

data_left_hand = pickle.load(open("../Data_preprocessing/flip_data.pickle", 'rb'))
data_left_hand = data_left_hand
data_left_hand_for_test = data_left_hand

label_counts = Counter(data_right_hand["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)
print()

label_counts = Counter(data_left_hand["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)
print()

def resampling_data(data):
    # Resampling data
    for label in range(28):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) > 4500 and len(filtered_data) < 5000:
            resampled_indices = np.random.choice(filtered_indices, size=5000 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (5000 - len(filtered_data)))
        
        if len(filtered_data) > 3500 and len(filtered_data) < 4000:  
            resampled_indices = np.random.choice(filtered_indices, size=4000 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (4000 - len(filtered_data)))

        if len(filtered_data) < 3500:
            resampled_indices = np.random.choice(filtered_indices, size=3500 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (3500 - len(filtered_data)))
    
    for label in range(28, 39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) > 1500 and len(filtered_data) < 1700:   
            resampled_indices = np.random.choice(filtered_indices, size=1700 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1700 - len(filtered_data)))

        if len(filtered_data) > 1000 and len(filtered_data) < 1500:      
            resampled_indices = np.random.choice(filtered_indices, size=1500 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1500 - len(filtered_data)))

        if str_label == '28':
            resampled_indices = np.random.choice(filtered_indices, size=300 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (300 - len(filtered_data)))

        if str_label == '33':
            resampled_indices = np.random.choice(filtered_indices, size=1800 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1800 - len(filtered_data)))

        if str_label == '38':
            resampled_indices = np.random.choice(filtered_indices, size=1000 - len(filtered_data), replace=True)
            resampled_data = [data['data'][idx] for idx in resampled_indices]

            data['data'].extend(resampled_data)
            data['labels'].extend([str_label] * (1000 - len(filtered_data)))

    label_counts = Counter(data["labels"])
    print(label_counts)

    return data


def augmentation_implementation(data):
    
    data = resampling_data(data)
    # Perform transformation by rotating each coordinate axis
    rotate_data = []
    rotate_labels = []

    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) == 5000:
            data_rotate_number = 500
        elif len(filtered_data) == 4000:
            data_rotate_number = 400
        elif len(filtered_data) == 3500:
            data_rotate_number = 450
        elif len(filtered_data) == 1700 or len(filtered_data) == 1500:
            data_rotate_number = 150
        else:
            data_rotate_number = 50

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))  
        data_rotate_final = []

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, 10, 'x')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
        
        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, -10, 'x')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, 10, 'y')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, -10, 'y')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
    
        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, 10, 'z')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            data_rotate = Augmentation.change_camera_perspective(divided_samples, -10, 'z')
            data_rotate_final.append(data_rotate)

        rotate_data.extend(data_rotate_final)
        rotate_labels.extend([str_label] * len(data_rotate_final))

    # Adjust the height and width of the hand
    change_data = []
    change_labels = []
    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]
        # print(len(filtered_data))

        if len(filtered_data) == 5000:
            data_change_ratio_number = 500
        elif len(filtered_data) == 4000:
            data_change_ratio_number = 400
        elif len(filtered_data) == 3500:
            data_change_ratio_number = 450
        elif len(filtered_data) == 1700 or len(filtered_data) == 1500:
            data_change_ratio_number = 150
        else:
            data_change_ratio_number = 50

        selected_change_samples = random.sample(filtered_data, min(data_change_ratio_number, len(filtered_data)))
        data_change_ratio_final = []

        for i in range(len(selected_change_samples)):
            divided_samples = selected_change_samples[i]

            ratio_x = random.uniform(0.95, 0.99)
            ratio_y = random.uniform(0.95, 0.99)

            data_change_ratio = Augmentation.change_hand_ratio(divided_samples, ratio_x, ratio_y)
            data_change_ratio_final.append(data_change_ratio)

        selected_change_samples = random.sample(filtered_data, min(data_change_ratio_number, len(filtered_data)))

        for i in range(len(selected_change_samples)):
            divided_samples = selected_change_samples[i]

            ratio_x_negative = random.uniform(1.01, 1.05)
            ratio_y_negative = random.uniform(1.01, 1.05)

            data_change_ratio = Augmentation.change_hand_ratio(divided_samples, ratio_x_negative, ratio_y_negative)
            data_change_ratio_final.append(data_change_ratio)   

        change_data.extend(data_change_ratio_final)
        change_labels.extend([str_label] * len(data_change_ratio_final))

    # Introduce minor fluctuations into the coordinate values of the landmarks
    add_minor_data = []
    add_minor_labels = []

    for label in range(39):
        str_label = str(label)
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        if len(filtered_data) == 5000:
            data_add_minor_number = 500
        elif len(filtered_data) == 4000:
            data_add_minor_number = 400
        elif len(filtered_data) == 3500:
            data_add_minor_number = 450
        elif len(filtered_data) == 1700 or len(filtered_data) == 1500:
            data_add_minor_number = 150
        else:
            data_add_minor_number = 50

        selected_add_minor_samples = random.sample(filtered_data, min(data_add_minor_number, len(filtered_data)))

        data_add_minor_final = []

        for i in range(len(selected_add_minor_samples)):
            divided_samples = selected_add_minor_samples[i]

            x_magnitude = 0.005
            y_magnitude = 0.005
            z_percentage = 0.02

            data_add_minor = Augmentation.add_minor_fluctuations(divided_samples, x_magnitude, y_magnitude, z_percentage)
            data_add_minor_final.append(data_add_minor)

        selected_add_minor_samples = random.sample(filtered_data, min(data_add_minor_number, len(filtered_data)))
        
        for i in range(len(selected_add_minor_samples)):
            divided_samples = selected_add_minor_samples[i]

            x_magnitude = 0.004
            y_magnitude = 0.004
            z_percentage = 0.01

            data_add_minor = Augmentation.add_minor_fluctuations(divided_samples, x_magnitude, y_magnitude, z_percentage)
            data_add_minor_final.append(data_add_minor)

        add_minor_data.extend(data_add_minor_final)
        add_minor_labels.extend([str_label] * len(data_add_minor_final))

    data['data'].extend(rotate_data)
    data['labels'].extend(rotate_labels)

    data['data'].extend(change_data)
    data['labels'].extend(change_labels)

    data['data'].extend(add_minor_data)
    data['labels'].extend(add_minor_labels)

    return data

right_hand_data = augmentation_implementation(data_right_hand)
right_hand_path = "../Data_preprocessing/right_hand_data.pickle"  
with open(right_hand_path, 'wb') as file:
    pickle.dump(right_hand_data, file)

left_hand_data = augmentation_implementation(data_left_hand)
left_hand_path = "../Data_preprocessing/left_hand_data.pickle"  
with open(left_hand_path, 'wb') as file:
    pickle.dump(left_hand_data, file)

label_counts = Counter(right_hand_data["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print("right hand dataset:", sorted_labels)

label_counts = Counter(left_hand_data["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print("left hand dataset:", sorted_labels)


# Create test data
def augmentation_implementation_create_testdata(data):

    raw_data = []
    raw_labels = []
    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        data_rotate_number = 200

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))        
        raw_data.extend(selected_samples)
        raw_labels.extend([str_label] * len(selected_samples))

    # Perform transformation by rotating each coordinate axis
    rotate_data = []
    rotate_labels = []

    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        data_rotate_number = 200
        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data))) 
        data_rotate_final = []

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            angles = np.random.randint(5, 31)
            data_rotate = Augmentation.change_camera_perspective(divided_samples, angles, 'x')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
        
        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            angles = np.random.randint(-30, -4)
            data_rotate = Augmentation.change_camera_perspective(divided_samples, angles, 'x')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            angles = np.random.randint(5, 31)
            data_rotate = Augmentation.change_camera_perspective(divided_samples, angles, 'y')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            angles = np.random.randint(-30, -4)
            data_rotate = Augmentation.change_camera_perspective(divided_samples, angles, 'y')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))
    
        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            angles = np.random.randint(5, 31)
            data_rotate = Augmentation.change_camera_perspective(divided_samples, angles, 'z')
            data_rotate_final.append(data_rotate)

        selected_samples = random.sample(filtered_data, min(data_rotate_number, len(filtered_data)))

        for i in range(len(selected_samples)):
            divided_samples = selected_samples[i]
            angles = np.random.randint(-30, -4)
            data_rotate = Augmentation.change_camera_perspective(divided_samples, angles, 'z')
            data_rotate_final.append(data_rotate)

        rotate_data.extend(data_rotate_final)
        rotate_labels.extend([str_label] * len(data_rotate_final))

    # Adjust the height and width of the hand
    change_data = []
    change_labels = []
    for label in range(39):
        str_label = str(label)
        
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]
        # print(len(filtered_data))

        data_change_ratio_number = 200
        selected_change_samples = random.sample(filtered_data, min(data_change_ratio_number, len(filtered_data)))
        data_change_ratio_final = []

        for i in range(len(selected_change_samples)):
            divided_samples = selected_change_samples[i]

            ratio_x = random.uniform(0.9, 0.99)
            ratio_y = random.uniform(0.9, 0.99)

            data_change_ratio = Augmentation.change_hand_ratio(divided_samples, ratio_x, ratio_y)
            data_change_ratio_final.append(data_change_ratio)

        selected_change_samples = random.sample(filtered_data, min(data_change_ratio_number, len(filtered_data)))

        for i in range(len(selected_change_samples)):
            divided_samples = selected_change_samples[i]

            ratio_x_negative = random.uniform(1.01, 1.1)
            ratio_y_negative = random.uniform(1.01, 1.1)

            data_change_ratio = Augmentation.change_hand_ratio(divided_samples, ratio_x_negative, ratio_y_negative)
            data_change_ratio_final.append(data_change_ratio)   

        change_data.extend(data_change_ratio_final)
        change_labels.extend([str_label] * len(data_change_ratio_final))

    # Introduce minor fluctuations into the coordinate values of the landmarks
    add_minor_data = []
    add_minor_labels = []

    for label in range(39):
        str_label = str(label)
        filtered_indices = [idx for idx, lbl in enumerate(data['labels']) if lbl == str_label]
        filtered_data = [data['data'][idx] for idx in filtered_indices]

        data_add_minor_number = 200
        selected_add_minor_samples = random.sample(filtered_data, min(data_add_minor_number, len(filtered_data)))
        data_add_minor_final = []

        for i in range(len(selected_add_minor_samples)):
            divided_samples = selected_add_minor_samples[i]

            x_magnitude = 0.008
            y_magnitude = 0.008
            z_percentage = np.random.uniform(0.01, 0.03)

            data_add_minor = Augmentation.add_minor_fluctuations(divided_samples, x_magnitude, y_magnitude, z_percentage)
            data_add_minor_final.append(data_add_minor)

        selected_add_minor_samples = random.sample(filtered_data, min(data_add_minor_number, len(filtered_data)))
        
        for i in range(len(selected_add_minor_samples)):
            divided_samples = selected_add_minor_samples[i]

            x_magnitude = 0.007
            y_magnitude = 0.007
            z_percentage = np.random.uniform(0.01, 0.03)

            data_add_minor = Augmentation.add_minor_fluctuations(divided_samples, x_magnitude, y_magnitude, z_percentage)
            data_add_minor_final.append(data_add_minor)

        add_minor_data.extend(data_add_minor_final)
        add_minor_labels.extend([str_label] * len(data_add_minor_final))
    
    data_test = {'data': [], 'labels': []}

    data_test['data'].extend(raw_data)
    data_test['labels'].extend(raw_labels)

    data_test['data'].extend(rotate_data)
    data_test['labels'].extend(rotate_labels)

    data_test['data'].extend(change_data)
    data_test['labels'].extend(change_labels)

    data_test['data'].extend(add_minor_data)
    data_test['labels'].extend(add_minor_labels)

    return data_test


right_hand_test_data = augmentation_implementation_create_testdata(data_right_hand_for_test)
right_hand_test_path = "../Data_preprocessing/right_hand_test_data.pickle"  
with open(right_hand_test_path, 'wb') as file:
    pickle.dump(right_hand_test_data, file)

left_hand_test_data = augmentation_implementation_create_testdata(data_left_hand_for_test)
left_hand_test_path = "../Data_preprocessing/left_hand_test_data.pickle"  
with open(left_hand_test_path, 'wb') as file:
    pickle.dump(left_hand_test_data, file)


label_counts = Counter(right_hand_test_data["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)

label_counts = Counter(left_hand_test_data["labels"])
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])
print(sorted_labels)