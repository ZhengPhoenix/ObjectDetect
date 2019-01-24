~/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
    --input_file=$(pwd)/squeezenet_model.pb \
    --input_format=TENSORFLOW_GRAPHDEF \
    --output_format=TFLITE \
    --output_file=$(pwd)/squeezenet_model.tflite \
    --inference_type=FLOAT \
    --input_type=FLOAT \
    --input_arrays=input_1 \
    --output_arrays=dense_1/BiasAdd \
    --input_shapes=1,227,227,3
