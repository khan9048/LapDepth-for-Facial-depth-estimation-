#An Open3D RGBDImage is composed of two images, RGBDImage.depth and RGBDImage.color.
# We require the two images to be registered into the same camera frame and have the same resolution.
# The following example show how to read and use RGBD images for point cloud.
import open3d as o3d
import matplotlib.pyplot as plt
print("Read Syn dataset")
#The format of stored depth is a 16-bit single channel image. The integer value represents the depth measurement in millimeters.
# It is the default format for Open3D to parse depth images.
color_raw = o3d.io.read_image(r"facial_depth_estimation_paper_resylts\bts\test\bathroom\rgb_0018.jpg")
depth_raw = o3d.io.read_image(r"facial_depth_estimation_paper_resylts\bts\test\bathroom\sync_depth_0018.png")
def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(1.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

print(rgbd_image)
#The default conversion function create_rgbd_image_from_color_and_depth creates an RGBDImage from a pair of color and depth image.
# The color image is converted into a grayscale image, stored in float ranged in [0, 1].
# The depth image is stored in float, representing the depth value in meters.

#The converted images can be rendered as numpy arrays.
plt.subplot(1, 2, 1)
plt.title('Syn grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Syn depth image')
plt.imshow(rgbd_image.depth)
plt.show()
import numpy as np
#####################################################################################################
# Camera calibration
# fov = 36
# w = 480
# h = 640
# cx, cy = 240, 320  # Principal point (cx, cy): The optical centre of the image plane
# fx, fy = 20, 20  # Focal length (fx, fy): measure the position of the image plane wrt to the camera centre.
#
# intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
# intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# cam = o3d.camera.PinholeCameraParameters()
# cam.intrinsic = intrinsic
#
# cam.extrinsic = np.array([[1.14893617e+03, 0.00000000e+00, 6.40000000e+02, 6.40000000e+02], [0.00000000e+00, 1.15038462e+03, 4.80000000e+02, 6.40000000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 6.40000000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 6.40000000e+02]])
#
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image, cam.intrinsic, cam.extrinsic)
#######################################################################################################################
#The RGBD image can be converted into a point cloud, given a set of camera parameters.
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


# Flip it, otherwise the pointcloud will be upside down
#pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
o3d.visualization.draw_geometries([pcd])
custom_draw_geometry_with_rotation(pcd)









