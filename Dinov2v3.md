## support Dinov2 Dinov3 backbone
1，新增Dinov2和Dinov3 backbone，像swin一样，适配数据集youtubevis，包含2019和2021的数据集；
mask2former和mask2former_video都需要支持Dinov2和Dinov3 backbone；
总的要求就是mask2former和mask2former_video都支持Dinov2和Dinov3像swin一样使用;
https://github.com/facebookresearch/dinov2；
https://github.com/facebookresearch/dinov3；
'''
Implemented new D2Dinov2 and D2Dinov3 backbones that load pretrained DINO vision transformers via torch.hub and expose their patch embeddings as Detectron2-compatible feature maps

Added configuration options for DINOv2/DINOv3 backbones and updated dependencies accordingly

Introduced YouTubeVIS 2019/2021 training configs for both DINOv2 and DINOv3 backbones, enabling dataset support similar to existing Swin configurations
'''
2，使用预训练好的dinvov3模型做推理测试，下载哪一个模型？如何用预训练好的模型做推理？
swin我本地测试可以这样：
python demo.py --config-file ../configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml   --input /home/luolu/Pictures/7.jpg  --output output  --opts MODEL.WEIGHTS /home/luolu/ModelPretrain/model_final_c5c739.pkl；
3，使用预训练好的dinvov2 ViT-g模型做推理测试;如何用预训练好的模型做推理？
比如swin我本地测试可以这样：
python demo.py --config-file ../configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml   --input /home/luolu/Pictures/7.jpg  --output output  --opts MODEL.WEIGHTS /home/luolu/ModelPretrain/model_final_c5c739.pkl；
## demo inference dinov2_vitg14
python demo.py   --config-file ../configs/youtubevis_2019/dinov2/video_maskformer2_dinov2_bs16_8ep.yaml   --input /home/luolu/Pictures/7.jpg   --output output

## dinov2 vitg14 training
python train_net_video.py --num-gpus 2 \
  --config-file configs/youtubevis_2019/dinov2/video_maskformer2_dinov2_bs16_8ep.yaml