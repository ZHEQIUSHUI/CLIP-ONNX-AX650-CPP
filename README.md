# CLIP


https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP/assets/46700201/7fefc9dd-9168-462d-bae9-bb013731f5c6


other interesting project [SAM-ONNX-AX650-CPP](https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP)

## Build
```
mkdir build
cd build
```
if x86 onnxruntime
```
cmake -DONNXRUNTIME_DIR=${onnxruntime_dir} -DOpenCV_DIR=${opencv_cmake_file_dir} ..
```
else if ax650
```
cmake -DONNXRUNTIME_DIR=${onnxruntime_dir} -DOpenCV_DIR=${opencv_cmake_file_dir} -DBSP_MSP_DIR=${msp_out_dir} -DBUILD_WITH_AX650=ON ..
```
```
make -j4
```
aarch64-none-gnu library:\
[onnxruntime](https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP/releases/download/ax_models/onnxruntime-aarch64-none-gnu-1.16.0.zip)\
[opencv](https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP/releases/download/ax_models/libopencv-4.6-aarch64-none.zip)

## Resource
[Google Drive](https://drive.google.com/drive/folders/13fAprRaBqoY-_c6CQroHnEoZ5iMQ0Cq_?usp=sharing)

## ONNX
### Export Onnx

[ZHEQIUSHUI/CLIP](https://github.com/ZHEQIUSHUI/CLIP)\
[ZHEQIUSHUI/Chinese-CLIP](https://github.com/ZHEQIUSHUI/Chinese-CLIP/tree/ax650)

### Get Original model
#### export onnx by yourself
```
# Original Clip
git clone https://github.com/ZHEQIUSHUI/CLIP.git
cd CLIP
python onnx_export.py

# Chinese Clip
git clone https://github.com/ZHEQIUSHUI/Chinese-CLIP.git
git checkout ax650

# download weights
cd weights
./downloads.sh

# get onnx model
cd ..
./convert.sh

# onnxsim model
cd ax650
./onnxsim.sh
```
or direct download model from release
```
# Chinese Clip model
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/cnclip/cnclip_vitb16.axmodel
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/cnclip/cnclip_vitb16.img.fp32.onnx
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/cnclip/cnclip_vitb16.txt.fp32.onnx

# feature matmul model
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/3models/feature_matmul.onnx

# Original Clip model
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/3models/image_encoder.onnx
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/3models/image_encoder.axmodel
wget https://github.com/ZHEQIUSHUI/CLIP-ONNX-AX650-CPP/releases/download/3models/text_encoder.onnx

```


### run in x86 with onnxruntime
#### 英文
```
./main --ienc image_encoder.onnx --tenc text_encoder.onnx --dec feature_matmul.onnx -v ../vocab.txt -i ../images/ -t ../text.txt 

inputs: 
              images: 1 x 3 x 224 x 224
output: 
      image_features: 1 x 512
decode Inference Cost time : 0.00040005s

per image:
                 image path\text|                            bird|                             cat|                             dog|
              ../images/bird.jpg|                            1.00|                            0.00|                            0.00|
               ../images/cat.jpg|                            0.00|                            0.99|                            0.01|
         ../images/dog-chai.jpeg|                            0.00|                            0.00|                            1.00|


per text:
                 text\image path|              ../images/bird.jpg|               ../images/cat.jpg|         ../images/dog-chai.jpeg|
                            bird|                            0.87|                            0.01|                            0.12|
                             cat|                            0.00|                            0.98|                            0.02|
                             dog|                            0.00|                            0.00|                            1.00|
```
#### 中文
```
./main -l 1 -v ../cn_vocab.txt -t ../cn_text.txt -i ../images/ --ienc ../onnx_models/vitb16.img.fp32.onnx --tenc ../onnx_models/vitb16.txt.fp32.onnx -d ../onnx_models/feature_matmul.onnx 

inputs: 
               image: 1 x 3 x 224 x 224
output: 
unnorm_image_features: 1 x 512
[I][              load_image_encoder][  20]: image feature len 512
[I][               load_text_encoder][ 101]: text feature len 512
[I][                  load_tokenizer][  75]: text token len 52
encode text Inference Cost time : 0.0926369s
matmul Inference Cost time : 0.00045888s

per image:
                 image path\text|                          小鸟|                          猫咪|                          狗子|
              ../images/bird.jpg|                            1.00|                            0.00|                            0.00|
               ../images/cat.jpg|                            0.00|                            0.99|                            0.01|
         ../images/dog-chai.jpeg|                            0.00|                            0.00|                            1.00|


per text:
                 text\image path|              ../images/bird.jpg|               ../images/cat.jpg|         ../images/dog-chai.jpeg|
                          小鸟|                            0.77|                            0.22|                            0.01|
                          猫咪|                            0.00|                            1.00|                            0.00|
                          狗子|                            0.00|                            0.00|                            1.00|
```
#### 中英混合
```
./main -l 1 -v ../cn_vocab.txt -t ../cn_text_mix.txt -i ../images/ --ienc ../onnx_models/vitb16.img.fp32.onnx --tenc ../onnx_models/vitb16.txt.fp32.onnx -d ../onnx_models/feature_matmul.onnx 

inputs: 
               image: 1 x 3 x 224 x 224
output: 
unnorm_image_features: 1 x 512
[I][              load_image_encoder][  20]: image feature len 512
[I][               load_text_encoder][ 101]: text feature len 512
[I][                  load_tokenizer][  75]: text token len 52
encode text Inference Cost time : 0.106218s
matmul Inference Cost time : 0.000361136s

per image:
                 image path\text|                        小 bird|                         cat 咪|                     小 dog 子|
              ../images/bird.jpg|                           1.00|                           0.00|                         0.00|
               ../images/cat.jpg|                           0.00|                           0.95|                         0.05|
         ../images/dog-chai.jpeg|                           0.00|                           0.01|                         0.99|


per text:
                 text\image path|              ../images/bird.jpg|               ../images/cat.jpg|         ../images/dog-chai.jpeg|
                         小 bird|                            0.96|                            0.03|                            0.00|
                          cat 咪|                            0.00|                            0.93|                            0.07|
                       小 dog 子|                            0.00|                            0.01|                            0.99|
```

## AX650
### run in AXERA Chip AX650 
#### 英文
```
./main --ienc image_encoder.axmodel --tenc text_encoder.onnx -d feature_matmul.onnx  -v vocab.txt -t text.txt -i images/
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done.
Engine alloc io is done.
[I][                            init][ 275]: RGB MODEL
decode Inference Cost time : 0.000754583s

per image:
                 image path\text|                            bird|                             cat|                             dog|
                 images/bird.jpg|                            1.00|                            0.00|                            0.00|
                  images/cat.jpg|                            0.01|                            0.98|                            0.01|
            images/dog-chai.jpeg|                            0.00|                            0.00|                            1.00|


per text:
                 text\image path|                 images/bird.jpg|                  images/cat.jpg|            images/dog-chai.jpeg|
                            bird|                            1.00|                            0.00|                            0.00|
                             cat|                            0.00|                            0.99|                            0.01|
                             dog|                            0.00|                            0.00|                            1.00|

```
#### 中文
```
./main -l 1 -v cn_vocab.txt -t cn_text.txt  -i images/ --ienc cn_clip_vitb16.axmodel --tenc vitb16.txt.fp32.onnx -d feature_matmul.onnx
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done.
Engine alloc io is done.
[I][                            init][ 275]: RGB MODEL
[I][              load_image_encoder][  19]: image feature len 512
[I][               load_text_encoder][ 101]: text feature len 512
[I][                  load_tokenizer][  75]: text token len 52
encode text Inference Cost time : 0.762541s
matmul Inference Cost time : 0.0007695s

per image:
                 image path\text|                            小鸟|                             猫咪|                            狗子|
                 images/bird.jpg|                            0.99|                            0.00|                            0.01|
                  images/cat.jpg|                            0.00|                            0.98|                            0.02|
            images/dog-chai.jpeg|                            0.00|                            0.00|                            1.00|


per text:
                 text\image path|                 images/bird.jpg|                  images/cat.jpg|            images/dog-chai.jpeg|
                           小鸟|                             0.43|                            0.57|                            0.00|
                           猫咪|                             0.00|                            1.00|                            0.00|
                           狗子|                             0.00|                            0.14|                            0.86|
```
#### 中英混和
```
./main -l 1 -v cn_vocab.txt -t cn_text_mix.txt  -i images/ --ienc cn_clip_vitb16.axmodel --tenc vitb16.txt.fp32.onnx -d feature_matmul.onnx
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done.
Engine alloc io is done.
[I][                            init][ 275]: RGB MODEL
[I][              load_image_encoder][  19]: image feature len 512
[I][               load_text_encoder][ 101]: text feature len 512
[I][                  load_tokenizer][  75]: text token len 52
encode text Inference Cost time : 0.75124s
matmul Inference Cost time : 0.000727667s

per image:
                 image path\text|                         小 bird|                          cat 咪|                        小 dog 子|
                 images/bird.jpg|                            0.99|                            0.01|                            0.00|
                  images/cat.jpg|                            0.00|                            0.94|                            0.06|
            images/dog-chai.jpeg|                            0.00|                            0.00|                            1.00|


per text:
                 text\image path|                 images/bird.jpg|                  images/cat.jpg|            images/dog-chai.jpeg|
                        小 bird|                             0.92|                            0.08|                            0.00|
                         cat 咪|                             0.00|                            1.00|                            0.00|
                      小 dog 子|                             0.00|                            0.10|                            0.90|
```
## Reference
[CLIP](https://github.com/openai/CLIP)\
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)\
[CLIP-ImageSearch-NCNN](https://github.com/EdVince/CLIP-ImageSearch-NCNN)
