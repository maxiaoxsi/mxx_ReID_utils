import torch
import numpy as np
import cv2
from .utils.visualization_utils import render_mesh, vis_keypoints_with_skeleton
from .utils.visualization_utils import perspective_projection
import os

class Painter:
    def __init__(self, path_smplx_model):
        import smplx
        self._model = smplx.create(
            path_smplx_model,
            model_type='smplx',
            gender='neutral', 
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='npz',
            use_pca=False
        )
        self._faces = self._model.faces 

        self._kps_lines = [
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

    def __call__(
        self, 
        img, 
        path_skeleton=None, 
        path_manikin=None, 
        is_save_skeleton = False, 
        is_save_manikin=False,
        is_read=True
    ):
        path_img = img.get_path('reid')
        path_smplx_pred = img.get_path('smplx_pred')
        if path_skeleton is None:
            path_skeleton = img.get_path('smplx_skeleton')
        if path_manikin is None:
            path_manikin = img.get_path('smplx_manikin')

        if is_read:
            if os.path.exists(path_skeleton) and os.path.exists(path_manikin):
                img_manikin = cv2.imread(path_manikin)
                img_skeleton = cv2.imread(path_skeleton)
                return img_manikin, img_skeleton
        if not os.path.exists(path_smplx_pred):
            return
        data_load = np.load(path_smplx_pred)

        betas = torch.from_numpy(data_load['smplx_shape']).float()      # [1, num_betas]
        expression = torch.from_numpy(data_load['smplx_expr']).float()  # [1, num_expression_coeffs]
        global_orient = torch.from_numpy(data_load['smplx_root_pose']).float()  # [1, 3]
        body_pose = torch.from_numpy(data_load['smplx_body_pose']).unsqueeze(0).float() # [1, 21*3] 或 [1, 21, 3]
        print(body_pose.shape)
        print(global_orient.shape)
        exit()

        left_hand_pose=torch.from_numpy(data_load['smplx_lhand_pose']).unsqueeze(0).float()
        right_hand_pose=torch.from_numpy(data_load['smplx_rhand_pose']).unsqueeze(0).float()
        jaw_pose=torch.from_numpy(data_load['smplx_jaw_pose']).float()
        leye_pose=torch.from_numpy(data_load['smplx_leye_pose']).float()
        reye_pose=torch.from_numpy(data_load['smplx_reye_pose']).float()

        transl = torch.from_numpy(data_load['cam_trans']).float()

        focal = data_load['focal'].tolist()
        princpt = data_load['princpt'].tolist()
        
        output = self._model(
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

        vertices = output.vertices.detach().cpu().numpy().squeeze()

        img = cv2.imread(path_img)
        img_black = img.copy()
        img_black[:, :, :] = 0
        img_manikin=render_mesh(img_black, vertices, self._faces, {'focal': focal, 'princpt':princpt}, mesh_as_vertices=False)
        if is_save_manikin:
            cv2.imwrite(path_manikin, img_manikin[:, :, ::-1])

        joints_3d = output.joints.detach().cpu().numpy().squeeze()
        joints_2d = perspective_projection(joints_3d, {'focal': focal, 'princpt':princpt})
        kps = np.zeros((3, joints_2d.shape[0]))
        kps[0, :] = joints_2d[:, 0]
        kps[1, :] = joints_2d[:, 1]
        kps[2, :] = 1  # 全部可见
        img_skeleton = vis_keypoints_with_skeleton(img_black, kps, self._kps_lines, kp_thresh=0.4, alpha=1)
        if is_save_skeleton:
            cv2.imwrite(path_skeleton, img_skeleton)
        return img_manikin, img_skeleton

