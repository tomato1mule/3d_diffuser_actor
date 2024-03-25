import os
import numpy as np
import open3d as o3d


root = 'visualization'

for task_str in os.listdir(root):
    for variation in os.listdir(os.path.join(root, task_str)):
        for demo_id in os.listdir(os.path.join(root, task_str, variation)):
            ep_root = os.path.join(root, task_str, variation, demo_id)
            print(ep_root)
            for p in sorted(os.listdir(ep_root), reverse=False):
                if 'rgb' in p:
                    rgb = np.load(os.path.join(ep_root, p))
                    pcd = np.load(os.path.join(ep_root, p.replace('rgb', 'pcd')))
                    gripper = np.load(os.path.join(ep_root, p.replace('rgb', 'gripper')))
                    action = np.load(os.path.join(ep_root, p.replace('rgb', 'action')))

                    # visualization
                    pcd_scene = o3d.geometry.PointCloud()
                    pcd_scene.colors = o3d.utility.Vector3dVector(rgb)
                    pcd_scene.points = o3d.utility.Vector3dVector(pcd)

                    coord_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    coord_gripper.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(gripper[3:3+4][[3,0,1,2]]))
                    coord_gripper.translate(gripper[:3])
                    sphere_gripper = o3d.geometry.TriangleMesh.create_sphere(0.02)
                    sphere_gripper.translate(gripper[:3])
                    sphere_gripper.paint_uniform_color([1 - gripper[-1], 0, gripper[-1]])

                    coord_action = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    coord_action.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(action[3:3 + 4][[3,0,1,2]]))
                    coord_action.translate(action[:3])
                    sphere_action = o3d.geometry.TriangleMesh.create_sphere(0.02)
                    sphere_action.translate(action[:3])
                    sphere_action.paint_uniform_color([1 - action[-1], 0, action[-1]])

                    world_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
                    o3d.visualization.draw_geometries([
                        world_coordinate,
                        pcd_scene,
                        coord_gripper, sphere_gripper,
                        coord_action, sphere_action,
                    ])