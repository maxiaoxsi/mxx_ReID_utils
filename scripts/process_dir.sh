DIR_REF=/machangxiao/datasets/ReID_ref
DIR_REID_SMPLX_MARKET=/machangxiao/datasets/ReID_smplx/Market-1501-v15.09.15/bounding_box_test

python run/process_dir.py \
    --dir_base $DIR_REF \
    --type_process delete \
    --name 170001_250712 \
    --name_new pred