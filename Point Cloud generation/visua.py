# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("s111.ply")
    o3d.visualization.draw_geometries([pcd])