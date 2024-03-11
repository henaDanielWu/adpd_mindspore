import mindspore

import _init_paths
import dataset
import torchvision.transforms as transforms
import torch
# from mindspore.dataset import GeneratorDataset 
# import mindspore.dataset
# import mindspore.dataset.vision as vision
# import mindspore.dataset.transforms as transforms
def dataload(cfg):
    # normalize = vision.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],is_hwc=False
    # )
    # valid_dataset = dataset.mpii(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         vision.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_loader =GeneratorDataset(
    #     valid_dataset,
    #     column_names =['input','target', 'target_weight', 'meta'],
    #     shuffle=False,
    #     num_parallel_workers=1
    # )
    # valid_loader=valid_loader.batch(32)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = dataset.mpii(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS
    )
    return valid_dataset,valid_loader
    
    