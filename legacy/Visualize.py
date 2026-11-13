import cv2
import numpy as np

# Function to extract RGB front camera image and camera calibration
def extract_front_camera_image(frame):
    # Extract camera and calibration from frame
    camera_name = dataset_pb2.CameraName.FRONT
    camera = waymo_utils.get(frame.images, camera_name)

    # Get image and convert to RGB
    image = waymo_utils.decode_image(camera)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# Function to visualize object labels in camera image
def project_labels_into_camera(camera_calibration, image, labels, labels_valid, img_resize_factor=1.0):
    # Get transformation matrix from vehicle frame to image
    vehicle_to_image = waymo_utils.get_image_transform(camera_calibration)

    # Draw all valid labels
    for label, vis in zip(labels, labels_valid):
        if vis:
            colour = (0, 255, 0)  # Green for valid labels
        else:
            colour = (255, 0, 0)  # Red for invalid labels

        # Only show labels of type "vehicle"
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            waymo_utils.draw_3d_box(image, vehicle_to_image, label, colour=colour)

    # Resize image if needed
    if img_resize_factor < 1.0:
        width = int(image.shape[1] * img_resize_factor)
        height = int(image.shape[0] * img_resize_factor)
        dim = (width, height)
        img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return img_resized
    else:
        return image

# Assuming we have the required data structures like `camera_calibration`, `image`, `object_labels`, `labels_valid`
# Project object labels into camera image
img_with_labels = project_labels_into_camera(camera_calibration, image, object_labels, labels_valid)

# Display the image with object labels
cv2.imshow('Object Labels in Camera Image', img_with_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()
