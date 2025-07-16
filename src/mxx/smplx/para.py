import torch
import numpy as np

def get_params_smplx(path_pred):
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

def combine_smplx_params(params_shape, params_pose):
    params_combined = {}
    params_combined['betas'] = params_shape['betas']          # 体型
    params_combined['expression'] = params_shape['expression'] # 表情
    
    # 从参数a获取动作和视角相关参数
    params_combined['transl'] = params_pose['transl']        # 位置/平移
    params_combined['global_orient'] = params_pose['global_orient']  # 全局旋转/视角
    params_combined['body_pose'] = params_pose['body_pose']         # 身体姿势
    params_combined['left_hand_pose'] = params_pose['left_hand_pose'] # 左手姿势
    params_combined['right_hand_pose'] = params_pose['right_hand_pose'] # 右手姿势
    params_combined['jaw_pose'] = params_pose['jaw_pose']           # 下巴姿势
    params_combined['leye_pose'] = params_pose['leye_pose']         # 左眼姿势
    params_combined['reye_pose'] = params_pose['reye_pose']         # 右眼姿势
    
    # 相机参数通常跟随视角(从a获取)或可以单独设置
    params_combined['focal'] = params_pose['focal']
    params_combined['princpt'] = params_pose['princpt']
    
    return params_combined