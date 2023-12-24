import numpy as np

# Changing the perspective using a 3D rotation matrix.
def change_camera_perspective(coordinates, angle, types):
    reshaped_coordinates = np.array(coordinates).reshape(-1, 3)
    
    # Perform transformation by rotating each coordinate axis
    if types == 'x':
        transformed_coordinates = []
        for group in reshaped_coordinates:
            transformed_group = change_camera_perspective_for_x_axes(group, angle)
            transformed_coordinates.extend(transformed_group)
    
    if types == 'y':
        transformed_coordinates = []
        for group in reshaped_coordinates:
            transformed_group = change_camera_perspective_for_y_axes(group, angle)
            transformed_coordinates.extend(transformed_group)

    if types == 'z':
        transformed_coordinates = []
        for group in reshaped_coordinates:
            transformed_group = change_camera_perspective_for_z_axes(group, angle)
            transformed_coordinates.extend(transformed_group)

    return transformed_coordinates

def change_camera_perspective_for_x_axes(coordinates, angle):
    radian_angle = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                 [0, np.cos(radian_angle), -np.sin(radian_angle)],
                                 [0, np.sin(radian_angle), np.cos(radian_angle)]])
    rotated_coordinates = np.dot(coordinates, rotation_matrix.T)
    
    return rotated_coordinates.tolist()

def change_camera_perspective_for_y_axes(coordinates, angle):
    radian_angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(radian_angle), 0, np.sin(radian_angle)],
                                 [0, 1, 0],
                                 [-np.sin(radian_angle), 0, np.cos(radian_angle)]])
    rotated_coordinates = np.dot(coordinates, rotation_matrix.T)

    return rotated_coordinates.tolist()

def change_camera_perspective_for_z_axes(coordinates, angle):
    radian_angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(radian_angle), -np.sin(radian_angle), 0],
                                 [np.sin(radian_angle), np.cos(radian_angle), 0],
                                 [0, 0, 1]])
    rotated_coordinates = np.dot(coordinates, rotation_matrix.T)
    return rotated_coordinates.tolist()



# Adjust the height and width of the hand
def change_hand_ratio(coordinates, width_ratio, height_ratio):
    reshaped_coordinates = np.array(coordinates).reshape(-1, 3)

    width = np.max(reshaped_coordinates[:, 0]) - np.min(reshaped_coordinates[:, 0])
    height = np.max(reshaped_coordinates[:, 1]) - np.min(reshaped_coordinates[:, 1])

    new_width = width * width_ratio
    new_height = height * height_ratio
    width_scale = new_width / width
    height_scale = new_height / height

    centered_coordinates = reshaped_coordinates - np.mean(reshaped_coordinates, axis=0)
    scaled_coordinates = centered_coordinates * np.array([width_scale, height_scale, 1.0])
    transformed_coordinates = scaled_coordinates + np.mean(reshaped_coordinates, axis=0)

    transformed_coordinates = transformed_coordinates.flatten().tolist()

    return transformed_coordinates
    

# Introduce minor fluctuations into the coordinate values of the landmarks
def add_minor_fluctuations(coordinates, x_magnitude, y_magnitude, z_percentage):
    reshaped_coordinates = np.array(coordinates).reshape(-1, 3)
    random_fluctuations_x = np.random.uniform(-x_magnitude, x_magnitude, size=(reshaped_coordinates.shape[0], 1))
    random_fluctuations_y = np.random.uniform(-y_magnitude, y_magnitude, size=(reshaped_coordinates.shape[0], 1))
    random_fluctuations_z = reshaped_coordinates[:, 2] * z_percentage

    coordinates_with_fluctuations = np.column_stack((reshaped_coordinates[:, 0] + random_fluctuations_x.flatten(),
                                                    reshaped_coordinates[:, 1] + random_fluctuations_y.flatten(),
                                                    reshaped_coordinates[:, 2] + random_fluctuations_z))
    coordinates_with_fluctuations = coordinates_with_fluctuations.flatten().tolist()

    return coordinates_with_fluctuations