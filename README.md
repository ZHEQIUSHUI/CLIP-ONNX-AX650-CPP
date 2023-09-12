# CLIP

other 因垂丝汀 project [SAM-ONNX-AX650-CPP](https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP)

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



## ONNX
### get onnx model
```
git clone https://github.com/ZHEQIUSHUI/CLIP.git
cd CLIP
python onnx_export.py
```
### 润 in x86 with onnxruntime
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

## AX650
### 润 in AXERA Chip AX650 
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

## Reference
[CLIP](https://github.com/openai/CLIP)\
[CLIP-ImageSearch-NCNN](https://github.com/EdVince/CLIP-ImageSearch-NCNN)