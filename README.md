# CLIP

## ONNX

```
./main -e ../onnx_models/CLIP_encoder.onnx -d ../onnx_models/decode.onnx -v ../vocab.txt -i ../images/ -t ../text.txt 

inputs: 
             input.1: 1 x 3 x 224 x 224
output: 
                2270: 1 x 512

per image:
                 image path\text|                            bird|                             cat|                             dog|
              ../images/bird.jpg|                            1.00|                            0.00|                            0.00|
               ../images/cat.jpg|                            0.00|                            0.99|                            0.01|
         ../images/dog-chai.jpeg|                            0.00|                            0.02|                            0.98|


per text:
                 text\image path|              ../images/bird.jpg|               ../images/cat.jpg|         ../images/dog-chai.jpeg|
                            bird|                            0.96|                            0.01|                            0.03|
                             cat|                            0.00|                            0.91|                            0.09|
                             dog|                            0.00|                            0.00|                            1.00|
```

# AX650

```
/opt/test/clip # ./main -e compiled.axmodel -d onnx_models/decode.onnx -v vocab.
txt -i images/ -t text.txt
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done.
Engine alloc io is done.
[I][                            init][ 275]: RGB MODEL

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