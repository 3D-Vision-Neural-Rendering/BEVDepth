import os
import argparse
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from multiprocessing import Process, Queue
import pickle as pkl
from tqdm import tqdm
from pdb import set_trace

cam_locations = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


def get_lidar_depth(lidar_points, img, lidar_info, cam_info):
    lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
    lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
    cam_calibrated_sensor = cam_info['calibrated_sensor']
    cam_ego_pose = cam_info['ego_pose']
    pts_img, depth = map_pointcloud_to_image(
        lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
        lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
    return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                            axis=1).astype(np.float32)

def load_lidar(data_root, lidar_infos):
    lidar_path = lidar_infos['LIDAR_TOP']['filename']
    lidar_points = np.fromfile(os.path.join(
        data_root, lidar_path),
                                dtype=np.float32,
                                count=-1).reshape(-1, 5)[..., :4]
    return lidar_points

def fun(data_root, annos):
    for anno in tqdm(annos):
        lidar_infos = anno['lidar_infos']
        cam_infos = anno['cam_infos']
        lidar_points = load_lidar(data_root, lidar_infos,)
        for cam_location in cam_locations:
            cam_info = cam_infos[cam_location]
            img = Image.open(
                    os.path.join(data_root, cam_info['filename']))
            point_depth = get_lidar_depth(
                        lidar_points, img,
                        lidar_infos, cam_info)
            depth_path = os.path.join(data_root,cam_info['filename'].replace('samples/','depths/').replace('.jpg','.npy') )
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            np.save(depth_path, point_depth)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='generate lidar depth')
    parse.add_argument('--root_dir', type=str, help='dataset root')
    parse.add_argument('--pkl_file', type=str, help='pkl file path')
    parse.add_argument('--proc_num', type=int, help='process num')
    args = parse.parse_args()

    process = []
    q = Queue()

    infos = pkl.load(open(args.pkl_file,'rb'))
    # fun(args.root_dir,infos)

    chunk_size = len(infos) // args.proc_num
    chunk_infos = [infos[i:i+chunk_size] for i in range(0, len(infos), chunk_size)]

    for chunk_info in chunk_infos: 
        p = Process(target=fun, args=(args.root_dir,chunk_info ))
        p.start()
        process.append(p)

    for p in process:
        p.join()

'''
python gen_lidar_depth.py \
--root_dir /media/data/mingshan/dataset/nuScenes \
--pkl_file  /media/data/mingshan/dataset/nuScenes/nuscenes_infos_train.pkl \
--proc_num 20 

'''
