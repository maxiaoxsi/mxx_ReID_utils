from mxx.utils.batch import process_batch
from mxx.utils.batch import rename_guidance

# from mxx.utils.batch import annot_drn_smplx, annot_is_backpack


# def annot_is_smplx_batch(path_cfg):
#     process_batch(
#         path_cfg=path_cfg, 
#         name_proc="annot_is_smplx", 
#         method_proc_dir=None,
#         method_proc_file=annot_is_smplx,    
#     )

def annot_upper_vl_batch(path_cfg):
    from mxx.annot.utils.batch import annot_upper_vl
    process_batch(
        path_cfg=path_cfg, 
        name_proc="annot_upper_vl", 
        method=annot_upper_vl,  
        batch_size=128,
        max_workers=4,
    )

def render_skeleton_batch(path_cfg):
    from mxx.smplx.utils.batch import render_skeleton
    process_batch(
        path_cfg=path_cfg, 
        name_proc="render_skeleton", 
        method=render_skeleton,  
        batch_size=128,
        max_workers=4,
    )

def annot_drn_smplx_batch(path_cfg):
    process_batch(
        path_cfg=path_cfg, 
        name_proc="annot_drn_smplx", 
        method_proc_dir=None,
        method_proc_file=annot_drn_smplx,    
    )

def rename_guidance_batch(path_cfg):
    process_batch(
        path_cfg=path_cfg, 
        name_proc="rename_guidance", 
        method_proc_dir=rename_guidance,
        method_proc_file=None,     
    )

def make_mask(path_cfg):
    from mxx.utils.batch import process_make_mask
    process_make_mask(path_cfg)

if __name__ == '__main__':
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/cfg_cache_market_train.yaml'
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/cfg_cache_msmt17_train.yaml'
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/cfg_cache_duke_train.yaml'
    path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml'
    # annot_is_smplx_batch(path_cfg)
    # rename_guidance_batch(path_cfg)
    # annot_drn_smplx_batch(path_cfg)
    # annot_is_backpack_batch(path_cfg)
    # render_skeleton(path_cfg)
    # make_mask(path_cfg)
    # annot_upper_vl_batch(path_cfg)
    # render_skeleton_batch(path_cfg=path_cfg)
    annot_upper_vl_batch(path_cfg)
    

