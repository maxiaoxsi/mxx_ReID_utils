import torch
import numpy as np
import cv2
from .utils.visualization_utils import render_mesh, vis_keypoints_with_skeleton
from .utils.visualization_utils import perspective_projection
import os
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

def get_smplx_data(path_pred):
    data_load = np.load(path_pred)
    ans = {}
    ans['betas'] = torch.from_numpy(data_load['smplx_shape']).float()      # [1, num_betas]
    ans['expression'] = torch.from_numpy(data_load['smplx_expr']).float()  # [1, num_expression_coeffs]
    ans['global_orient'] = torch.from_numpy(data_load['smplx_root_pose']).float()  # [1, 3]
    ans['body_pose'] = torch.from_numpy(data_load['smplx_body_pose']).unsqueeze(0).float() # [1, 21*3] 或 [1, 21, 3]
    ans['left_hand_pose']=torch.from_numpy(data_load['smplx_lhand_pose']).unsqueeze(0).float()
    ans['right_hand_pose']=torch.from_numpy(data_load['smplx_rhand_pose']).unsqueeze(0).float()
    ans['jaw_pose']=torch.from_numpy(data_load['smplx_jaw_pose']).float()
    ans['leye_pose']=torch.from_numpy(data_load['smplx_leye_pose']).float()
    ans['reye_pose']=torch.from_numpy(data_load['smplx_reye_pose']).float()
    ans['transl'] = torch.from_numpy(data_load['cam_trans']).float()
    ans['focal'] = data_load['focal'].tolist()
    ans['princpt'] = data_load['princpt'].tolist()
    return ans

 # 使用四元数球面线性插值处理全局旋转
def slerp_rotation(rot1, rot2, alpha):
    """使用四元数slerp进行旋转插值"""
    # 将pytorch tensor转换为numpy并展平
    rot1_np = rot1.detach().cpu().numpy().flatten()
    rot2_np = rot2.detach().cpu().numpy().flatten()
    
    # 将旋转向量转换为Rotation对象
    r1 = Rotation.from_rotvec(rot1_np)
    r2 = Rotation.from_rotvec(rot2_np)
    
    # 检查四元数方向一致性，确保选择最短路径
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    
    # 如果四元数点积为负，翻转其中一个以确保最短路径插值
    if np.dot(q1, q2) < 0:
        q2 = -q2
        r2 = Rotation.from_quat(q2)
    
    # 创建slerp插值器
    key_rots = Rotation.from_quat([q1, q2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    
    # 进行插值
    interpolated_rot = slerp([alpha])[0]
    
    # 转换回旋转向量格式并转为tensor
    result_rotvec = interpolated_rot.as_rotvec()
    return torch.from_numpy(result_rotvec).float().reshape(rot1.shape)

def interpolate_smplx_params(para1, para2, alpha):
    """
    在两个SMPL-X参数之间进行插值
    alpha: 插值系数，0表示para1，1表示para2
    使用四元数slerp进行旋转插值，其他参数线性插值
    """
    
    interpolated = {}
    
    # 自动识别并对所有张量参数进行线性插值（除了全局旋转）
    for key in para1.keys():
        # if key in ['focal', 'princpt']:
        #     # 相机参数保持第一个不变
        #     interpolated[key] = para1[key]
        if key == 'global_orient':
            # global_orient 使用slerp处理，稍后单独处理
            pass
        elif isinstance(para1[key], torch.Tensor):
            # 对所有其他张量参数进行线性插值
            interpolated[key] = para1[key] * (1 - alpha) + para2[key] * alpha
        else:
            # 非张量参数保持第一个不变
            interpolated[key] = para1[key]

    # 处理全局旋转
    global_orient1 = para1['global_orient']
    global_orient2 = para2['global_orient']
    
    # 使用slerp进行旋转插值
    interpolated['global_orient'] = slerp_rotation(global_orient1, global_orient2, alpha)
    
    return interpolated
  


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
        img=None,
        path_pred_ref=None,
        path_pred_tgt=None,
        path_img=None, 
        path_skeleton=None, 
        path_manikin=None, 
        is_save_skeleton = False, 
        is_save_manikin=False,
        is_read=True
    ):
        # path_img = img.get_path('reid')
        # path_pred = img.get_path('smplx_pred')
        # if path_skeleton is None:
        #     path_skeleton = img.get_path('smplx_skeleton')
        # if path_manikin is None:
        #     path_manikin = img.get_path('smplx_manikin')

        # if is_read:
        #     if os.path.exists(path_skeleton) and os.path.exists(path_manikin):
        #         img_manikin = cv2.imread(path_manikin)
        #         img_skeleton = cv2.imread(path_skeleton)
        #         return img_manikin, img_skeleton
        
        
        if not os.path.exists(path_pred_ref):
            return
        
        if not os.path.exists(path_pred_tgt):
            return
        
        para_smplx_ref = get_smplx_data(path_pred_ref)
        para_smplx_tgt = get_smplx_data(path_pred_tgt)
        
        # 生成插值帧序列
        num_frames = 12  # 增加帧数以包含起始和结束帧
        interpolated_frames = []
        
        for i in range(num_frames):
            alpha = i / (num_frames - 1)  # 从0到1的插值系数
            
            # 边界条件：直接使用原始参数避免数值误差
            if alpha == 0.0:
                interpolated_frames.append(para_smplx_ref)
            elif alpha == 1.0:
                interpolated_frames.append(para_smplx_tgt)
            else:
                interpolated_params = interpolate_smplx_params(para_smplx_ref, para_smplx_tgt, alpha)
                interpolated_frames.append(interpolated_params)
        # 渲染所有插值帧
        img = cv2.imread(path_img)
        img_black = img.copy()
        img_black[:, :, :] = 0
        
        # 存储所有帧图像用于生成GIF
        gif_frames = []
        
        for frame_idx, current_frame in enumerate(interpolated_frames):
            print(f"正在渲染第 {frame_idx + 1}/{num_frames} 帧...")
            
            # 获取当前帧的SMPL-X模型输出
            output = self._model(
                betas=current_frame['betas'], 
                expression=current_frame['expression'],
                body_pose=current_frame['body_pose'],
                global_orient=current_frame['global_orient'],
                left_hand_pose=current_frame['left_hand_pose'],
                right_hand_pose=current_frame['right_hand_pose'],
                jaw_pose=current_frame['jaw_pose'],
                leye_pose=current_frame['leye_pose'],
                reye_pose=current_frame['reye_pose'],
                transl=current_frame['transl'],
                return_verts=True
            )

            vertices = output.vertices.detach().cpu().numpy().squeeze()
            
            # 渲染当前帧
            img_manikin = render_mesh(img_black, vertices, self._faces, 
                                    {'focal': current_frame['focal'], 'princpt': current_frame['princpt']}, 
                                    mesh_as_vertices=False)
            
            # 根据渲染结果调整图像方向
            # 请根据实际渲染效果选择合适的翻转方式：
            # img_manikin_flipped = cv2.flip(img_manikin, 0)   # 垂直翻转（上下翻转）
            img_manikin_flipped = cv2.flip(img_manikin, 1)   # 水平翻转（左右翻转）
            # img_manikin_flipped = cv2.flip(img_manikin, -1)  # 既垂直又水平翻转（180度旋转）
            # img_manikin_flipped = cv2.flip(img_manikin, -1)  # 尝试180度旋转
            
            # 保存当前帧PNG（使用翻转后的图像）
            frame_filename = f'./interpolated_frame_{frame_idx:02d}.png'
            cv2.imwrite(frame_filename, img_manikin_flipped[:, :, ::-1])
            print(f"已保存: interpolated_frame_{frame_idx:02d}.png")
            
            # 将帧添加到GIF列表中 (注意opencv读取的是BGR，需要转换为RGB)
            img_rgb = cv2.cvtColor(img_manikin_flipped, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            gif_frames.append(pil_img)
        
        # 生成GIF动画
        print("正在生成GIF动画...")
        if gif_frames:
            # 保存为GIF，设置循环播放和帧间隔
            gif_frames[0].save(
                './smplx_interpolation.gif',
                save_all=True,
                append_images=gif_frames[1:],
                duration=200,  # 每帧持续200ms
                loop=0  # 无限循环
            )
            print("GIF动画已保存: smplx_interpolation.gif")
            
            # 生成来回播放的GIF（更平滑的动画效果）
            gif_frames_loop = gif_frames + gif_frames[-2:0:-1]  # 添加反向帧
            gif_frames_loop[0].save(
                './smplx_interpolation_loop.gif',
                save_all=True,
                append_images=gif_frames_loop[1:],
                duration=200,
                loop=0
            )
            print("循环GIF动画已保存: smplx_interpolation_loop.gif")
        
        print(f"所有 {num_frames} 个插值帧已生成完成!")
        print("已生成两个GIF文件:")
        print("1. smplx_interpolation.gif - 单向播放")
        print("2. smplx_interpolation_loop.gif - 来回循环播放")
        exit()
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

