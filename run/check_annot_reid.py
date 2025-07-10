
import argparse
from mxx.utils.batch import process_reid_batch_vl, process_reid_batch
from mxx.annot import get_arg_bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_cfg", type=str, default="/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/cfg_cache_mars_train.yaml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--is_riding", type=str, default="False")
    parser.add_argument("--is_backpack", type=str, default="False")
    parser.add_argument("--is_shoulder_bag", type=str, default="False")
    parser.add_argument("--is_hand_carried", type=str, default="False")
    parser.add_argument("--color_upper", type=str, default="False")
    parser.add_argument("--color_bottoms", type=str, default="False")
    args = parser.parse_args()
    path_cfg = args.path_cfg
    batch_size = args.batch_size
    max_workers = args.max_workers

    if get_arg_bool(args.is_riding):
        from mxx.ReID.annot.check_annot import check_is_riding
        print("check is_riding")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="check is_riding",
            method=check_is_riding,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    if get_arg_bool(args.is_backpack):
        from mxx.ReID.annot.check_annot import check_is_backpack
        print("check is_backpack")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="check is_backpack",
            method=check_is_backpack,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    if get_arg_bool(args.is_shoulder_bag):
        from mxx.ReID.annot.check_annot import check_is_shoulder_bag
        print("check is_shoulder_bag")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="check is_shoulder_bag",
            method=check_is_shoulder_bag,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    if get_arg_bool(args.is_hand_carried):
        from mxx.ReID.annot.check_annot import check_is_hand_carried
        print("check is_hand_carried")
        process_reid_batch(
            path_cfg=path_cfg,
            name_processing="check is_hand_carried",
            method=check_is_hand_carried,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    # if get_arg_bool(args.is_hand_carried):
    #     from mxx.ReID.annot.check_annot import check_color_upper
    #     print("check color_upper")
    #     process_reid_batch(
    #         path_cfg=path_cfg,
    #         name_processing="check color_upper",
    #         method=check_color_upper,
    #         batch_size=batch_size,
    #         max_workers=max_workers,
    #     )
