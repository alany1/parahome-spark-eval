import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import trange

from __init__ import *
from utils_headless import makeTpose, rotation_6d_to_matrix, BodyMaker, HandMaker, \
    get_stickman, get_stickhand, simpleViewerHeadless

SCAN_ROOT = os.path.join(ROOT_REPOSITORY, "data/scan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", default="data/seq/s1", help="target path, e.g : ")
    parser.add_argument("--start_frame", default=0, type=int, help="Render start frame number") 
    parser.add_argument("--end_frame", default=100000, type=int, help="Render end frame number")
    parser.add_argument("--run", default=False, action='store_true', help='If set, viewer will show start_frame scene at specific view')
    parser.add_argument("--fromort", default=False, action='store_true', help='Make skeleton from orientation, transform to ')
    parser.add_argument("--ego", default=True, action='store_true')

    args = parser.parse_args()
    
    root = os.path.join(ROOT_REPOSITORY, args.scene_root)
    camera_dir = root + "/cam_param"

    
    head_tip_position = pickle.load(open(root + "/head_tips.pkl", "rb"))
    if args.fromort:
        bone_vector = pickle.load(open(root + "/bone_vectors.pkl", "rb"))
        bodyTpose, HandTpose = makeTpose(bone_vector)
        
        body_ort = pickle.load(open(root + "/body_joint_orientations.pkl", "rb"))
        hand_ort = pickle.load(open(root + "/hand_joint_orientations.pkl", "rb"))
        body_rotmat = rotation_6d_to_matrix(torch.tensor(body_ort)).numpy()
        hand_rotmat = rotation_6d_to_matrix(torch.tensor(hand_ort)).numpy()        

        bodymaker = BodyMaker(bodyTpose)
        handmaker = HandMaker(HandTpose)
        body_joint = bodymaker(torch.tensor(body_ort)).numpy()
        hand_joint = handmaker(torch.tensor(hand_ort)).numpy()

        lhand = np.einsum('FMN, FJN -> FJM', body_rotmat[:,14], hand_joint[:,0]) + body_joint[:, 14][:,None,:]
        rhand = np.einsum('FMN, FJN -> FJM', body_rotmat[:,10], hand_joint[:,1]) + body_joint[:, 10][:,None,:]        
        joint_root_fixed = np.concatenate([body_joint, lhand, rhand], axis=1)

        body_global_trans = pickle.load(open(root + "/body_global_transform.pkl", "rb"))
        joint_rgb = np.einsum('FMN, FJN->FJM',body_global_trans[:,:3,:3], joint_root_fixed) + body_global_trans[:, :3, 3][:,None,:]
    else:
        joint_rgb = pickle.load(open(root + "/joint_positions.pkl", "rb"))
        
        if args.ego:
            body_global_trans = pickle.load(open(root + "/body_global_transform.pkl", "rb"))
            body_ort = pickle.load(open(root + "/body_joint_orientations.pkl", "rb"))
            body_rotmat = rotation_6d_to_matrix(torch.tensor(body_ort)).numpy()   


    
    frame_length = joint_rgb.shape[0]

    os.path.dirname(__file__)
    obj_color = json.load(open(os.path.join(ROOT_REPOSITORY,"visualize/color.json"), "r"))
    object_transform = pickle.load(open(root + "/object_transformations.pkl", "rb"))

    
    print("------Reading Objects------")
    initialized, mesh_dict = dict(), dict()
    obj_in_scene = json.load(open(Path(root, "object_in_scene.json"), "r"))
    # load_object
    for objn in obj_in_scene:
        for pn in ["base", "part1", "part2"]:
            meshpath = Path(SCAN_ROOT, objn, "simplified", pn+".obj")
            if meshpath.exists():
                keyn = objn + "_" + pn
                initialized[keyn] = False
                m = o3d.io.read_triangle_mesh(str(meshpath)) 
                m.paint_uniform_color(obj_color[objn][pn])
                m.compute_vertex_normals()
                mesh_dict[keyn] = m


    # make camera parameters
    width, height = 1600, 800
    if args.ego:
        view = o3d.camera.PinholeCameraParameters()
        camera_matrix = np.eye(3, dtype=np.float64)
        f = 520
        camera_matrix[0,0] = f
        camera_matrix[1,1] = f
        camera_matrix[0,2] = width/2
        camera_matrix[1,2] = height/2
        view.intrinsic.intrinsic_matrix = camera_matrix
        view.intrinsic.width, view.intrinsic.height = width, height
    else:
        view=None
    
    vis = simpleViewerHeadless("Render Scene", 1600, 800, [], view) # 
    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    vis.add_geometry({"name":"global", "geometry":global_coord})    
    vis.add_plane()
    renders = []
    depth_renders = []
    for fn in trange(args.start_frame, min(args.end_frame+1, frame_length), 10):
        """
        Head tip position 
        """
        if fn not in object_transform:
            continue

        cur_human_joints = joint_rgb[fn]
        
        cur_object_pose = object_transform[fn]
        
        bmesh = get_stickman(cur_human_joints[:23], head_tip_position[fn] if not args.ego else None)
        hmesh = get_stickhand(cur_human_joints[23:48]) + get_stickhand(cur_human_joints[48:])# hand
        bmesh.compute_vertex_normals()
        hmesh.compute_vertex_normals()
        
        vis.add_geometry({"name":"human", "geometry":bmesh+hmesh})

        # category 별로 나눠서 보기
        for inst_name, loaded in initialized.items():
            if loaded and inst_name in cur_object_pose:
                vis.transform(inst_name, cur_object_pose[inst_name])
            elif loaded and inst_name not in cur_object_pose: 
                vis.remove_geometry(inst_name)
                initialized[inst_name] = False
            elif not loaded and inst_name in cur_object_pose:
                vis.add_geometry({"name":inst_name, "geometry":mesh_dict[inst_name]})
                vis.transform(inst_name, cur_object_pose[inst_name])
                initialized[inst_name] = True
            elif not loaded and inst_name not in cur_object_pose:
                continue

        if args.ego:
            head_posinrgb = cur_human_joints[5]
            head_rotinrgb = body_global_trans[fn,:3,:3]@body_rotmat[fn, 6]
            # Change Axis Directions 
            head_rotinrgb = np.stack([-head_rotinrgb[:,1],-head_rotinrgb[:,2],head_rotinrgb[:,0]],axis=1)
            head_Trgb = np.eye(4, dtype=np.float64)
            head_Trgb[:3,:3] = head_rotinrgb
            head_Trgb[:3,3] = head_posinrgb
            extrinsic_matrix = np.linalg.inv(head_Trgb)
            vis.setupcamera(extrinsic_matrix)

        render = vis.grab_render("rgb", "depth")
        renders.append(render["rgb"])
        
        depth_viz = np.clip(render["depth"], 0, 5)
        depth_viz = ((depth_viz/5.0)*255).astype(np.uint8)
        depth_renders.append(depth_viz)
        
        vis.remove_geometry("human")

    from ml_logger import logger
    logger.configure(root="/home/exx/Downloads", prefix=".")
    with logger.Prefix("render"):
        logger.save_video(renders, "render.mp4", fps=10)
        logger.save_video(depth_renders, "depth_render.mp4", fps=10)
    



        


    
