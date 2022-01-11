# import pyvista as pv
# mesh = pv.read('sss.ply')
# cpos = mesh.plot()
import numpy as np
import open3d as o3d
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("s111.ply")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

# print("Downsample the point cloud with a voxel of 0.05")
# downpcd = pcd.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd])



