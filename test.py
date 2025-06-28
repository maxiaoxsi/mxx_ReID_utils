
from tqdm import tqdm


from mxx import ReIDProcessor, ReIDDataset

def init_prompts():
    prompts = {}

    prompts["bottoms_vl"] = "Could you please describe \
            the color and style of the bottoms of the people? \
            Answer with a brief phrase 'color' + 'style'\
            such as  'black jeans'. \
            A brief phrase is enough, no full sentences."
    
    prompts["upper_vl"] = "Could you please describe \
            the color and style of the upper clothing of the people? \
            Answer with a brief phrase 'color' + 'style'\
            such as  'white t-shirt'. \
            A brief phrase is enough, no full sentences."

    prompts["color_upper_vl"] = "Could you please describe \
        the color of the upper clothing from the text?\
        Answer with a brief phrase 'color' from phrase 'color' + 'style'\
        such as answer 'white' when the text is 'white t-shirt'. \
        A brief phrase is enough, no full sentences."

    prompts["color_bottoms_vl"] = "Could you please describe \
        the color of the bottoms from the text?\
        Answer with a brief phrase 'color' from phrase 'color' + 'style'\
        such as answer 'red' when the text is 'red shorts'. \
        A brief phrase is enough, no full sentences."

    prompts["is_shoulder_bag_vl"] = "Could you please describe \
        is the person in the photo carrying a shoulder bag?\
        please anwser with a brief phrase 'yes' or 'no'"
    
    prompts["is_hand_carried_vl"] = "Could you please describe \
        is the person in the photo carrying some object in his or her hand?\
        Attention that only the object carried on his hand,\
        the backpack and shoulder bag not count!\
        please anwser with a brief phrase 'yes' or 'no'"

    return prompts

def get_annots(processor, prompts):
    processor.get_annot_img(
        key_tgt="is_hand_carried_vl",
        prompt=prompts["is_hand_carried_vl"]
    )
    
    processor.get_annot_img(
        key_tgt="upper_vl",
        prompt=prompts["upper_vl"]
    )

    processor.get_annot_img(
        key_tgt="bottoms_vl",
        prompt=prompts["bottoms_vl"]
    )

    processor.get_annot_annot(
        key_annot_list=["upper_vl"],
        key_tgt="color_upper_vl",
        prompt=prompts["color_upper_vl"]
    )
    
    processor.get_annot_annot(
        key_annot_list = ["bottoms_vl"],
        key_tgt = "color_bottoms_vl",
        prompt=prompts["color_bottoms_vl"]
    )

    processor.get_annot_img(
        key_tgt="is_shoulder_bag_vl",
        prompt=prompts["is_shoulder_bag_vl"]
    )

if __name__ == '__main__':
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/MSMT17/humandataset_msmt17_train.yaml'
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/Market-1501-v15.09.15/humandataset_market_train.yaml'
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/DukeMTMC-reID/humandataset_duke_train.yaml'
    # path_cfg = '/machangxiao/code/MIP-ReID/configs/datasets/MARS-v160809/humandataset_mars_train.yaml'
    path_cfg = './humandataset_market_train.yaml'
   
    dataset_market = ReIDDataset(
        path_cfg=path_cfg,
        img_size_pad=(512, 512),
        n_img = (0, 100000),
        stage=1        
    )
    
    processor = ReIDProcessor(
        dataset=dataset_market,
        # is_vl=True
    )
    processor(
        'overwrite_key', 
        key='is_riding_vl',
        data_check = 'yes.', 
        data_new = 'yes'
    )
    exit()
    # processor.remove_key_annot("is_riding")
    
    # processor.overwrite_annot("is_riding_vl", 'm', 'yes')

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    prompts = init_prompts()
    
    

    get_annots(processor, prompts)
    
    processor.overwrite_annot("is_visible", 'visible', 'True')
    

    processor.rename_key_annot("vector_direction", "vec_drn_smplx")
    processor.rename_key_annot("mark_direction", "mark_drn_smplx")
    processor.rename_key_annot("direction", "drn_smplx")
    processor.rename_key_annot("direction_vl", "drn_vl")

    processor.rename_key_annot("riding_vl", "is_riding_vl")
    processor.rename_key_annot("backpack_vl", "is_backpack_vl")
    processor.rename_key_annot("glasses_vl", "is_glasses_vl")
    processor.rename_key_annot("visible", "is_visible")


    # processor.rename_key_annot("width", "width")
    # processor.rename_key_annot("height", "height")

    # processor.rename_key_annot("hand-carried_vl", "is_hand_carried_vl")
    

    # processor.rename_key_annot("color_upper_vl", "color_upper_vl")
    # processor.rename_key_annot("color_bottoms_vl", "color_bottoms_vl")
    # processor.overwrite_annot("is_hand_carried_vl", 'False', 'no')
    # processor.overwrite_annot("is_hand_carried_vl", 'True', 'yes')
    # processor.overwrite_annot("is_riding_vl", 'False', 'no')
    # processor.overwrite_annot("is_riding_vl", 'True', 'yes')
    # processor.overwrite_annot("is_backpack_vl", 'False', 'no')
    # processor.overwrite_annot("is_backpack_vl", 'True', 'yes')
    # processor.overwrite_annot("is_glasses_vl", 'False', 'no')
    # processor.overwrite_annot("is_glasses_vl", 'True', 'yes')








