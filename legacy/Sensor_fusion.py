import cv2
import open3d as o3d
import numpy as np

def process_kinect_data(rgb_image, depth_image):
    # Convert depth image to point cloud
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=525.0, fy=525.0, cx=319.5, cy=239.5  # Default Kinect intrinsics
    )
    depth_image = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        depth_image,
        depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic_matrix
    )

    # Optionally, perform filtering or downsampling on the point cloud

    return pcd

def fuse_sensor_data(kinect_point_cloud, inertial_data):
    # Perform fusion using Kalman Filter or other fusion algorithms
    # For demonstration purposes, let's just concatenate the inertial data to point cloud
    # Replace this with your fusion algorithm

    # Convert inertial data to 3D points
    inertial_points = np.zeros((len(inertial_data), 3))
    # Assume inertial_data is a list of [x, y, z] measurements
    inertial_points[:, :3] = inertial_data

    # Combine kinect_point_cloud and inertial_points
    fused_point_cloud = np.vstack((kinect_point_cloud.points, inertial_points))

    return fused_point_cloud

if __name__ == "__main__":
    # Load Kinect data (example)
    rgb_image = cv2.imread("rgb_image.png")
    depth_image = cv2.imread("depth_image.png", cv2.IMREAD_UNCHANGED)

    # Process Kinect data
    kinect_point_cloud = process_kinect_data(rgb_image, depth_image)

    # Load inertial sensor data (example)
    inertial_data = np.random.rand(100, 3)  # actual data

    # Fuse sensor data
    fused_point_cloud = fuse_sensor_data(kinect_point_cloud, inertial_data)

    # Visualize fused point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_point_cloud)
    o3d.visualization.draw_geometries([pcd])
