python optimize_for_inference.py \
--input=squeezenet_model.pb \
--output=squeezenet.pb \
--frozen_graph=True \
--input_names=input_1 \
--output_names=dense_1/BiasAdd
