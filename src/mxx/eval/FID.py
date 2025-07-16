import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights 
from scipy import linalg
from tqdm.auto import tqdm

# 计算激活统计量（均值和协方差矩阵）
def calculate_activation_statistics(images, model, device='cpu'):
    """计算图像的激活统计量"""
    batch_size = 64
    model.eval()
    act = []
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = []
        for img in images[i:i+batch_size]:
            img = Image.fromarray(img).convert('RGB')
            img = transform(img)
            batch_images.append(img)
        
        batch_images = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            pred = model(batch_images)
        
        # 将输出展平
        pred = pred.view(pred.size(0), -1).cpu().numpy()
        act.append(pred)
    
    act = np.vstack(act)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

# 计算FID分数
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个分布之间的Frechet距离"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, '两个均值向量维度必须相同'
    assert sigma1.shape == sigma2.shape, '两个协方差矩阵维度必须相同'
    
    diff = mu1 - mu2
    
    # 计算矩阵平方根
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 数值检查，确保实部为正
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 数值检查，确保实部为正
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('复数部分 {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def load_images_from_pathList(pathList, target_size=(299, 299)):
    images = []
    for path_img in pathList:
        try:
            img = Image.open(path_img).convert('RGB')
            img = img.resize(target_size)
            img = np.array(img)
            images.append(img) 
        except Exception as e:
            print(f"无法加载图像 {path_img}: {e}")
        
    return np.array(images)

def get_pathList_from_folder(dir_base, idx_poses):
    pathList = []
    for idx in idx_poses:
        dir = dir_base + str(idx)
        for root, dirs, files in os.walk(dir):
            for filename in files:
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    path_img = os.path.join(root, filename)
                    pathList.append(path_img)
    return pathList
            
# 创建InceptionV3特征提取模型
def create_inception_model(device='cpu'):
    """创建InceptionV3模型用于特征提取"""
    # 加载预训练的InceptionV3模型
    # inception = models.inception_v3(pretrained=True, weights=Inception_V3_Weights.DEFAULT)
    inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    # 修改模型以获取特征向量
    class InceptionFeatureExtractor(torch.nn.Module):
        def __init__(self, inception):
            super(InceptionFeatureExtractor, self).__init__()
            # 保留到最后一个卷积层的部分
            self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
            self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
            self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
            self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
            self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
            self.Mixed_5b = inception.Mixed_5b
            self.Mixed_5c = inception.Mixed_5c
            self.Mixed_5d = inception.Mixed_5d
            self.Mixed_6a = inception.Mixed_6a
            self.Mixed_6b = inception.Mixed_6b
            self.Mixed_6c = inception.Mixed_6c
            self.Mixed_6d = inception.Mixed_6d
            self.Mixed_6e = inception.Mixed_6e
            self.Mixed_7a = inception.Mixed_7a
            self.Mixed_7b = inception.Mixed_7b
            self.Mixed_7c = inception.Mixed_7c
            
            # 确保模型在评估模式下
            self.eval()
            
            # 冻结参数
            for param in self.parameters():
                param.requires_grad = False
        
        def forward(self, x):
            # 确保输入尺寸正确
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
            # 前向传播到特征提取层
            x = self.Conv2d_1a_3x3(x)
            x = self.Conv2d_2a_3x3(x)
            x = self.Conv2d_2b_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.Conv2d_3b_1x1(x)
            x = self.Conv2d_4a_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.Mixed_5b(x)
            x = self.Mixed_5c(x)
            x = self.Mixed_5d(x)
            x = self.Mixed_6a(x)
            x = self.Mixed_6b(x)
            x = self.Mixed_6c(x)
            x = self.Mixed_6d(x)
            x = self.Mixed_6e(x)
            x = self.Mixed_7a(x)
            x = self.Mixed_7b(x)
            x = self.Mixed_7c(x)
            
            # 全局平均池化
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            
            return x
    
    # 创建特征提取模型
    feature_extractor = InceptionFeatureExtractor(inception).to(device)
    return feature_extractor

def make_clipreid_model(cfg_clipreid):
    from clip_reid.model.make_model_clipreid_vmulti2 import make_model
    clip_reid = make_model(cfg_clipreid, num_class=1047, camera_num=6, view_num = 1)
    clip_reid.load_param(cfg_clipreid.TEST.WEIGHT)
    clip_reid = clip_reid.to(device="cuda")
    clip_reid.eval()
    return clip_reid  

# 主函数
def calculate_fid_score(imgList_reid, imgList_gen, device='cpu'):
    """计算生成图像与原图像间的FID分数"""
    # 设置设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        device = 'cpu'
    
    # 创建Inception模型
    print("加载Inception模型...")
    inception_model = create_inception_model(device)
    
    # 加载图像
    original_images = imgList_reid
    
    generated_images = imgList_gen
 
    # 计算统计量
    print("计算原始图像的激活统计量...")
    m1, s1 = calculate_activation_statistics(original_images, inception_model, device)
    
    print("计算生成图像的激活统计量...")
    m2, s2 = calculate_activation_statistics(generated_images, inception_model, device)
    
    # 计算FID
    print("计算FID分数...")
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value

def get_imgs_left():
    from mxx_processor import ReIDProcessor
    processor = ReIDProcessor(
        dir_reid='/machangxiao/datasets/ReID/Market-1501-v15.09.15/bounding_box_train',
        dir_gen='/machangxiao/datasets/ReID_gen/animateanyone/Market-1501-v15.09.15/pose0',
        dir_dscrpt='/machangxiao/datasets/ReID_dscrpt/Market-1501-v15.09.15/bounding_box_train'
    )
    pathList_reid = processor.get_pathList_by_direction("reid", "left")
    pathList_gen = processor.get_pathList("gen")
    return pathList_reid, pathList_gen

def get_imgs_by_direction(direction):
    from mxx_processor import ReIDProcessor
    processor = ReIDProcessor(
        dir_reid='/machangxiao/datasets/ReID/Market-1501-v15.09.15/bounding_box_train',
        dir_gen='/machangxiao/datasets/ReID_gen/animateanyone/Market-1501-v15.09.15/bounding_box_train/pose0',
        dir_dscrpt='/machangxiao/datasets/ReID_dscrpt/Market-1501-v15.09.15/bounding_box_train'
    )
    if direction == "all":
        pathList_reid = processor.get_pathList("reid")
    else:
        pathList_reid = processor.get_pathList_by_direction("reid", direction)
    
    pathList_gen = processor.get_pathList("gen")
    return pathList_reid, pathList_gen

def get_imgs(direction):
    # if direction == "right" or direction == "left" or direction == "fornt" or direction == "back":
    return get_imgs_by_direction(direction)
    

if __name__ == "__main__":
    pathList_reid, _ = get_imgs("all")
    print(len(pathList_reid))
    pathList_gen = get_pathList_from_folder(
        '/machangxiao/datasets/ReID_gen/animateanyone_ori/Market-1501-v15.09.15/bounding_box_train/pose', 
        [0, 2]
    )
    import random
    print(len(pathList_gen))
    pathList_gen = random.sample(pathList_gen, len(pathList_reid))
    print(len(pathList_gen))
    device = 'cuda'
    print(pathList_reid[0])
    print(pathList_gen[0])
    
    print("加载原始图像...")
    imgList_reid = load_images_from_pathList(pathList_reid)
    print(f"加载了 {len(imgList_reid)} 张原始图像")

    print("加载生成图像...")
    imgList_gen = load_images_from_pathList(pathList_gen)
    print(len(imgList_gen))
   
    print(f"加载了 {len(imgList_gen)} 张原始图像")
    # 计算FID

    fid_score = calculate_fid_score(imgList_reid, imgList_gen, device)
    print(f"FID分数: {fid_score}")    


    