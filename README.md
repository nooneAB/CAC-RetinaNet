Datasets  
====
The datasets used in this paper are DIOR and Levir-SHIP.  
DIOR: 
You can download it here:https://aistudio.baidu.com/datasetdetail/15179  
Levir-ship: 
You can download it here:https://github.com/WindVChen/Levir-Ship  
`The detectron2 framework in this project is for reference only, please go to the official website for specific use.`
https://detectron2-zhcn.readthedocs.io/zh-cn/latest/index.html  

Datasets format conversion
=
Because detectron2 supports coco formats, follow voc_format_coco.py  

COCO setting
=
You need to set up COCO following the official tutorial of Detectron2.  

Train
=
    python train_coco.py --config-file configs/coco/CSQ.train.yaml --num-gpu X OUTPUT_DIR work_dirs/coco_CSQ  

Test(The first is normal inference and the second is query inference)
=
    python infer_coco.py --config-file configs/coco/CSQ_test.yaml --num-gpu X --eval-only MODEL.WEIGHTS work_dirs/coco_CSQ/model_final.pth OUTPUT_DIR work_dirs/model_test    
    python infer_coco.py --config-file configs/coco/CSQ_test.yaml --num-gpu X --eval-only MODEL.WEIGHTS work_dirs/coco_CSQ/model_final.pth OUTPUT_DIR work_dirs/model_test MODEL.QUERY.QUERY_INFER True    

Citation
=
        @InProceedings{  
            author    = {{Jiazhen Li,Xuanhong Wang,Hongyu Guo,Xian Wang,Mingchen Wang}},  
            title     = {{Enhanced Small Target Detection in Optical Remote Sensing Images using CAC-RetinaNet with Multi-Scale Feature Fusion and Context Attention},  
            journal =   {The Visual Computer},  
            year      = {2025}  
        }  



