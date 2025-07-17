from omegaconf import OmegaConf
from mxx import ReIDDataset
from mxx.smplx import get_params_smplx, get_params_betas_mean, combine_smplx_params
from mxx.smplx import render_manikin
import os
import torch

def prepare_pose_std(dataset, dir_pose_std, dir_save):
    print(dataset.keys)
    print(dir_pose_std)
    print(dir_save)

    pose_std = {'front': {}, 'back': {}, 'left': {}, 'right': {}}
    for key in pose_std.keys():
        dir_drn = os.path.join(dir_pose_std, key)
        filename =  [filename  for filename in os.listdir(dir_drn) if 'npz' in filename][0]
        path_param = os.path.join(dir_drn, filename)
        pose_std[key]['path_param'] = path_param
        pose_std[key]['param'] = get_params_smplx(path_param)
        # print(pose_std[key]['path_params'])
        # print(pose_std[key]['params'].keys())
    
    for key in dataset.keys:
        person = dataset.get_person(key)
        params_person = []
        for basename_img in person.keys:
            img = person[basename_img]
            path_param = img.get_path("pred")
            if os.path.exists(path_param):
                params_person.append(get_params_smplx(path_param))
        betas_mean = get_params_betas_mean(params_person)
        param_shape = params_person[0]
        betas_mean = torch.from_numpy(betas_mean).unsqueeze(0).float()
        param_shape["betas"] = betas_mean
        path_img = img.get_path("reid")
        for drn in pose_std.keys():
            param_pose = pose_std[drn]['param']
            param = combine_smplx_params(
                params_shape = param_shape,
                params_pose = param_pose,
            )
            # print(param)
            print(path_img)
            path_manikin = os.path.join(dir_save, key, f"{drn}.jpg")
            render_manikin(param=param, path_img=path_img, path_manikin=path_manikin)
        # exit()
    # path_pred = get_path_pred()




if __name__ == "__main__":
    path_cfg_inf = '/machangxiao/code/MIP-ReID/configs/inference.yaml' 
    dir_pose_std = '/machangxiao/standard'
    dir_save = '/machangxiao/ReID_std'
    cfg = OmegaConf.load(path_cfg_inf)
    dataset = ReIDDataset(
        path_cfg=cfg.data.path_cfg['market'],
        img_size=(cfg.data.train_width, cfg.data.train_height),
        n_frame=cfg.data.n_frame,
        stage=cfg.data.stage,
        width_scale=(cfg.data.width_scale, 1),
        height_scale=(cfg.data.height_scale, 1),
        path_log=os.path.join(dir_save, "warning.log"),
        is_select_bernl=cfg.data.is_select_bernl,
        is_select_repeat=cfg.data.is_select_repeat,
        rate_random_erase=cfg.data.rate_random_erase,
        rate_dropout_ref=cfg.data.rate_dropout_ref,
        rate_dropout_back=cfg.data.rate_dropout_back,
        rate_dropout_manikin=cfg.data.rate_dropout_manikin,
        rate_dropout_skeleton=cfg.data.rate_dropout_skeleton,
        rate_dropout_rgbguid=cfg.data.rate_dropout_rgbguid,
        rate_mask_aug=cfg.data.rate_mask_aug,
    )
    prepare_pose_std(dataset, dir_pose_std, dir_save)