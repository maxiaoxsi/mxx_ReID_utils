
import argparse
from mxx.utils.batch import process_reid_batch_vl, process_reid_batch
from mxx.annot import get_arg_bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--is_smplx", type=str, default="False")
    parser.add_argument("--drn", type=str, default="False")
    parser.add_argument("--is_visible", type=str, default="False")
    parser.add_argument("--is_backpack", type=str, default="False")
    parser.add_argument("--is_shoulder_bag", type=str, default="False")
    parser.add_argument("--is_hand_carried", type=str, default="False")
    parser.add_argument("--upper", type=str, default="False")
    parser.add_argument("--color_upper", type=str, default="False")
    parser.add_argument("--style_upper", type=str, default="False")
    parser.add_argument("--bottoms", type=str, default="False")
    parser.add_argument("--color_bottoms", type=str, default="False")
    parser.add_argument("--style_bottoms", type=str, default="False")
    args = parser.parse_args()
    path_cfg = args.path_cfg
    batch_size = args.batch_size
    max_workers = args.max_workers

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
    exit()
    if get_arg_bool(args.is_backpack):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_backpack",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_backpack",
        )
    if get_arg_bool(args.is_shoulder_bag):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_shoulder_bag",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_shoulder_bag",
        )
    if get_arg_bool(args.is_hand_carried):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting is_hand_carried",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="is_hand_carried",
        )
    if get_arg_bool(args.upper):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting upper",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="upper",
        )
    if get_arg_bool(args.color_upper):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=['upper'],
            name_processing="annoting color_upper",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="color_upper",
        )
    if get_arg_bool(args.style_upper):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=['upper'],
            name_processing="annoting style_upper",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="style_upper",
        )
    if get_arg_bool(args.bottoms):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=[],
            name_processing="annoting bottoms",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="bottoms",
        )
    if get_arg_bool(args.color_bottoms):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=['bottoms'],
            name_processing="annoting color_bottoms",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="color_bottoms",
        )
    if get_arg_bool(args.style_bottoms):
        from mxx.annot.utils.annot_vl import annot_vl
        process_reid_batch_vl(
            path_cfg=path_cfg,
            keys_text=['bottoms'],
            name_processing="annoting style_bottoms",
            method_batch=annot_vl,
            batch_size=batch_size,
            idx_annot="style_bottoms",
        )
