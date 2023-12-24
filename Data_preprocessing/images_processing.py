import os
import pickle
import mediapipe as mp
import cv2
import shutil

def extract_hand_landmarks(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hands.close()
    return results.multi_hand_landmarks

def move_images_with_landmarks(input_directory, output_directory, limit_per_folder=5000):
    for root, dirs, files in os.walk(input_directory):
        for dir_name in dirs:
            source_directory = os.path.join(input_directory, dir_name)
            print(source_directory)
            destination_directory = os.path.join(output_directory, dir_name)
            print(destination_directory)

            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            image_files = [file for file in os.listdir(source_directory)]
            count = 0 
            for image_file in image_files:
                if count >= limit_per_folder:
                    break  
                source_image_path = os.path.join(source_directory, image_file)
                landmarks = extract_hand_landmarks(source_image_path)

                if landmarks:
                    destination_image_path = os.path.join(destination_directory, image_file)
                    shutil.copyfile(source_image_path, destination_image_path)
                    count += 1  

input_test_directory = "D:\\New_data_ASL\\asl_alphabet_train"
output_train_directory = "D:\\[COCA]_ASL_recognition\Model\\right_hand_images"

move_images_with_landmarks(input_test_directory, output_train_directory)


def flip_images(input_directory, output_directory):
    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            source_image_path = os.path.join(input_directory, file_name)
            destination_image_path = os.path.join(output_directory, file_name)

            img = cv2.imread(source_image_path)
            flipped_img = cv2.flip(img, 1) 
            cv2.imwrite(destination_image_path, flipped_img)

input_directory = "D:\\[COCA]_ASL_recognition\Model\\right_hand_images"
output_directory = "D:\\[COCA]_ASL_recognition\Model\\left_hand_images"

flip_images(input_directory, output_directory)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def create_dataset(img_path, pickle_path):
    data = []
    labels = []
    for dir_ in os.listdir(img_path):
        for img_path in os.listdir(os.path.join(img_path, dir_)):
            data_aux = []

            x_ = []
            y_ = []
            z_ = []

            img = cv2.imread(os.path.join(img_path, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z

                        x_.append(x)
                        y_.append(y)
                        z_.append(z)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = z_[i]

                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                        data_aux.append(z)

                data.append(data_aux)
                labels.append(dir_)

    f = open(pickle_path, 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

right_hand_img = "D:\\[COCA]_ASL_recognition\Model\\right_hand_images"
right_hand_data = "D:\\[COCA]_ASL_recognition\\Model\\right_hand_data.pickle"
lef_hand_img = "D:\\[COCA]_ASL_recognition\Model\\left_hand_images"
left_hand_data = "D:\\[COCA]_ASL_recognition\\Model\\left_hand_data.pickle"

create_dataset(right_hand_img, right_hand_data)
create_dataset(lef_hand_img, left_hand_data)