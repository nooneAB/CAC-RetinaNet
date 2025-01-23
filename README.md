Title
=

《Enhanced Small Target Detection in Optical Remote Sensing Images using CAC-RetinaNet with Multi-Scale Feature Fusion and Context Attention》    

journal
=

《The Visual Computer》  

Environment setting
=

Pytorch1.10.1+Cuda11.3+Nvidia 4090 24GB  
detectron2 is a core library that needs to be installed in your virtual environment, and the version we recommend here is 0.6.  
To install due to https://github.com/facebookresearch/detectron2/releases.

Doi
=
https://doi.org/10.5281/zenodo.14627521    

Datasets  
====
The datasets used in this paper are DIOR and Levir-SHIP.  
DIOR: 
You can download it here:https://aistudio.baidu.com/datasetdetail/15179  
Levir-ship: 
You can download it here:https://github.com/WindVChen/Levir-Ship  


Datasets format conversion
=
Because detectron2 supports coco formats, follow voc_format_coco.py  

COCO setting
=
You need to set up COCO following the official tutorial of Detectron2.  
https://detectron2-zhcn.readthedocs.io/zh-cn/latest/index.html  

Train
=
    python train_coco.py --config-file configs/coco/CSQ.train.yaml --num-gpu 1 OUTPUT_DIR work_dirs/coco_CSQ  

Test(The first is normal inference and the second is query inference)
=
    python infer_coco.py --config-file configs/coco/CSQ_test.yaml --num-gpu 1 --eval-only MODEL.WEIGHTS work_dirs/coco_CSQ/model_final.pth OUTPUT_DIR work_dirs/model_test    
    python infer_coco.py --config-file configs/coco/CSQ_test.yaml --num-gpu 1 --eval-only MODEL.WEIGHTS work_dirs/coco_CSQ/model_final.pth OUTPUT_DIR work_dirs/model_test MODEL.QUERY.QUERY_INFER True    

Citation
=
        @InProceedings{  
            author    = {{Jiazhen Li,Xuanhong Wang,Hongyu Guo,Xian Wang,Mingchen Wang}},  
            title     = {{Enhanced Small Target Detection in Optical Remote Sensing Images using CAC-RetinaNet with Multi-Scale Feature Fusion and Context Attention},  
            journal =   {The Visual Computer},  
            year      = {2025}  
        }  



