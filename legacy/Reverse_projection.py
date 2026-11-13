import cv2
import numpy as np

def reverse_project(depth_image, intrinsic_matrix):
    # Access camera intrinsics
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Create mesh grid of image coordinates
    rows, cols = depth_image.shape
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Convert image coordinates to homogeneous coordinates
    u_hom = np.stack((u, v, np.ones_like(u)), axis=-1)

    # Apply inverse of camera intrinsic matrix to get normalized coordinates
    K_inv = np.linalg.inv(intrinsic_matrix)
    normalized_coords = np.matmul(K_inv, u_hom.reshape(-1, 3).T).T
    
    # Reshape to original shape
    normalized_coords = normalized_coords.reshape(rows, cols, 3)

    # Apply depth information to get 3D points in camera coordinates
    points_3d_cam = normalized_coords * depth_image[..., np.newaxis]

    return points_3d_cam

# Example usage
if __name__ == "__main__":
    # Load depth image (example)
    depth_image = cv2.imread("depth_image.png", cv2.IMREAD_UNCHANGED)

    # Access camera intrinsics (example)
    intrinsic_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])

    # Perform reverse projection
    points_3d_cam = reverse_project(depth_image, intrinsic_matrix)

    # Visualize or further process the resulting 3D points

