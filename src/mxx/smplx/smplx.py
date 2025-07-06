import os
import smplx
import numpy as np
import torch
import cv2
from .utils.visualization_utils import render_mesh, vis_keypoints_with_skeleton
from .utils.visualization_utils import perspective_projection
from concurrent.futures import ProcessPoolExecutor
import glob
import errno

model_smplx = None
kps_lines = None

def load_smplx(path_smplx_model):
    model_smplx = smplx.create(
        path_smplx_model,
        model_type='smplx',
        gender='neutral', 
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz',
        use_pca=False
    ).to("cuda")

    kps_lines = [
        # 躯干
        [0, 3],   # pelvis - spine1
        [3, 6],   # spine1 - spine2
        [6, 9],   # spine2 - spine3
        [9, 12],  # spine3 - neck
        [12, 15], # neck - head

        # 左腿
        [0, 1],   # pelvis - left_hip
        [1, 4],   # left_hip - left_knee
        [4, 7],   # left_knee - left_ankle
        [7, 10],  # left_ankle - left_foot

        # 右腿
        [0, 2],   # pelvis - right_hip
        [2, 5],   # right_hip - right_knee
        [5, 8],   # right_knee - right_ankle
        [8, 11],  # right_ankle - right_foot

        # 左臂
        [12, 13], # neck - left_collar
        [13, 16], # left_collar - left_shoulder
        [16, 18], # left_shoulder - left_elbow
        [18, 20], # left_elbow - left_wrist

        # 右臂
        [12, 14], # neck - right_collar
        [14, 17], # right_collar - right_shoulder
        [17, 19], # right_shoulder - right_elbow
        [19, 21], # right_elbow - right_wrist
    ]
    return model_smplx, kps_lines


def render_skeleton(args):
    global model_smplx, kps_lines
    try:
        (
            path_img, 
            path_pred, 
            path_skeleton
        ) = args
        if model_smplx is None:
            model_smplx, kps_lines = load_smplx(
                path_smplx_model = '/machangxiao/code/smplx/models'
            )
        data_load = np.load(path_pred)
        betas = torch.from_numpy(data_load['smplx_shape']).float().to("cuda")      # [1, num_betas]
        expression = torch.from_numpy(data_load['smplx_expr']).float().to("cuda")  # [1, num_expression_coeffs]
        global_orient = torch.from_numpy(data_load['smplx_root_pose']).float().to("cuda")  # [1, 3]
        body_pose = torch.from_numpy(data_load['smplx_body_pose']).unsqueeze(0).float().to("cuda") # [1, 21*3] 或 [1, 21, 3]
        left_hand_pose=torch.from_numpy(data_load['smplx_lhand_pose']).unsqueeze(0).float().to("cuda")
        right_hand_pose=torch.from_numpy(data_load['smplx_rhand_pose']).unsqueeze(0).float().to("cuda")
        jaw_pose=torch.from_numpy(data_load['smplx_jaw_pose']).float().to("cuda")
        leye_pose=torch.from_numpy(data_load['smplx_leye_pose']).float().to("cuda")
        reye_pose=torch.from_numpy(data_load['smplx_reye_pose']).float().to("cuda")

        transl = torch.from_numpy(data_load['cam_trans']).float().to("cuda")

        focal = data_load['focal'].tolist()
        princpt = data_load['princpt'].tolist()
        
        output = model_smplx(
            betas=betas, 
            expression=expression,
            body_pose=body_pose,
            global_orient=global_orient,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            transl=transl,
            return_verts=True
        )
        img = cv2.imread(path_img)
        img_black = img.copy()
        img_black[:, :, :] = 0

        joints_3d = output.joints.detach().cpu().numpy().squeeze()
        joints_2d = perspective_projection(joints_3d, {'focal': focal, 'princpt':princpt})
        kps = np.zeros((3, joints_2d.shape[0]))
        kps[0, :] = joints_2d[:, 0]
        kps[1, :] = joints_2d[:, 1]
        kps[2, :] = 1  # 全部可见
        img_skeleton = vis_keypoints_with_skeleton(img_black, kps, kps_lines, kp_thresh=0.4, alpha=1)
        dir_skeleton = os.path.dirname(path_skeleton)
        try:
            os.makedirs(dir_skeleton)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        cv2.imwrite(path_skeleton, img_skeleton)
        return
    except Exception as e:
        import traceback
        with open('/tmp/render_skeleton_error.log', 'a') as f:
            f.write(traceback.format_exc() + '\n')
        print(traceback.format_exc())
        raise e
    


    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model_smplx.faces 
    img_manikin=render_mesh(img_black, vertices, faces, {'focal': focal, 'princpt':princpt}, mesh_as_vertices=False)
    if is_save_manikin:
        cv2.imwrite(path_manikin, img_manikin[:, :, ::-1])