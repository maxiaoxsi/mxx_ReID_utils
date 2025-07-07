import argparse
from mxx.utils.batch import process_batch_vl

def annot_vl_batch(path_cfg, batch_size, annot):
    from mxx.annot.utils.annot_vl import annot_vl
    process_batch_vl(
        path_cfg=path_cfg,
        name_batch=annot,
        method_batch=annot_vl,
        batch_size=batch_size,
        idx_annot=annot,
    )
    

# def annot_bottoms_vl_batch(path_cfg, batch_size):
#     from mxx.annot.utils.annot_vl import annot_vl
#     process_batch_vl(
#         path_cfg=path_cfg,
#         name_batch="annot_bottoms_vl",
#         method_batch=annot_vl,
#         batch_size=batch_size,
#         idx_annot="bottoms_vl"
#     )

# def annot_upper_vl_batch(path_cfg, batch_size):
#     from mxx.annot.utils.annot_vl import annot_upper_vl
#     process_batch_vl(
#         path_cfg=path_cfg, 
#         name_batch="annot_upper_vl", 
#         method_batch=annot_upper_vl,  
#         batch_size=batch_size,
#     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--annot", type=str, default="is_riding_vl")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    path_cfg = args.path_cfg
    annot = args.annot
    batch_size = args.batch_size
    annot_vl_batch(
        path_cfg=path_cfg, 
        batch_size=batch_size,
        annot=annot,
    )
    

