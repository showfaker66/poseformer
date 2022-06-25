# 基于改进Transformer的三维人体姿态估计

## 1、环境

python3.8 +cuda11.0+torch1.7.1

## 2、数据集

Human3.6M  localed in ./data directory.

## 3、Train

If CPN detected 2D pose as input: 

python run_poseformer.py -k cpn_ft_h36m_dbb

If Ground truth 2D pose as input:

python run_poseformer.py -k gt

### 4、Test 

训练好的权重放在./checkpoint directory

python run_poseformer.py  -c checkpoint --evaluate Your model

python run_poseformer.py  -c checkpoint --evaluate Your model

## 5、Visualization

python run_poseformer.py -k gt -arc 3,3,3,3 -c checkpoint --evaluate best_epoch.bin --render --viz-subject S11 --viz-action Walking --viz-camera 0 --viz-video "Videos/S11/Greeting 2.58860488.mp4" --viz-output figure/Purchases.gif/mp4 --viz-size 3 --viz-downsample 2 --viz-limit 150

截取视频帧，6s的视频，一共150帧，1秒25帧，一共截取150张图片

运行 ffmpeg -i figure/output.mp4 -ss 00:00 -t 6 -f image2 -vf fps=fps=25 figure/yiba_frame_%02d.png

