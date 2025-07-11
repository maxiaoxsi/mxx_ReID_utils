
import argparse
from mxx.utils.batch import process_reid_batch_vl, process_reid_batch
from mxx.annot import get_arg_bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--is_width", type=str, default="False")
    parser.add_argument("--is_smplx", type=str, default="False")
    parser.add_argument("--drn", type=str, default="False")
    parser.add_argument("--is_visible", type=str, default="False")
    parser.add_argument("--is_riding_vl", type=str, default="False")
    parser.add_argument("--is_backpack_vl", type=str, default="False")
    parser.add_argument("--is_shoulder_bag_vl", type=str, default="False")
    parser.add_argument("--is_hand_carried_vl", type=str, default="False")
    parser.add_argument("--upper_vl", type=str, default="False")
    parser.add_argument("--color_upper_vl", type=str, default="False")
    parser.add_argument("--style_upper_vl", type=str, default="False")
    parser.add_argument("--bottoms_vl", type=str, default="False")
    parser.add_argument("--color_bottoms_vl", type=str, default="False")
    parser.add_argument("--style_bottoms_vl", type=str, default="False")
    args = parser.parse_args()
    path_cfg = args.path_cfg
    batch_size = args.batch_size
    max_workers = args.max_workers

    if get_arg_bool(args.is_width):
        from mxx.annot.utils.annot_width import annot_width
        print("annot width start!")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="annoting width",
            method=annot_width,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    if get_arg_bool(args.is_smplx):
        from mxx.annot.utils.annot_is_smplx import annot_is_smplx
        print("annot reid start!")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="annoting is smplx",
            method=annot_is_smplx,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        

    if get_arg_bool(args.drn):
        from mxx.annot.utils.annot_drn import annot_drn
        print("annot drn start!")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="annoting drn",
            method=annot_drn,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        

    if get_arg_bool(args.is_visible):
        from mxx.annot.utils.annot_is_visible import annot_is_visible
        print("annot is_visible start!")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="annoting is_visible",
            method=annot_is_visible,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        

    if get_arg_bool(args.is_riding_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_riding_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_riding_vl",
        )
        

    if get_arg_bool(args.is_backpack_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_backpack_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_backpack_vl",
        )
        

    if get_arg_bool(args.is_shoulder_bag_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_shoulder_bag_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_shoulder_bag_vl",
        )
        

    if get_arg_bool(args.is_hand_carried_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_hand_carried_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_hand_carried_vl",
        )
        

    if get_arg_bool(args.upper_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting upper_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="upper_vl",
        )
        

    if get_arg_bool(args.color_upper_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=['upper'],
            name_processing="annoting color_upper_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="color_upper_vl",
        )
        

    if get_arg_bool(args.bottoms_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting bottoms_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="bottoms_vl",
        )
        

    if get_arg_bool(args.color_bottoms_vl):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=['bottoms'],
            name_processing="annoting color_bottoms_vl",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="color_bottoms_vl",
        )
        
