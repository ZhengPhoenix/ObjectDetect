python3 freeze_graph.py \
--input_meta_graph=squeezenet_model.ckpt.meta \
--input_checkpoint=squeezenet_model.ckpt \
--output_graph=squeezenet_model.pb \
--output_node_names="dense_1/BiasAdd" \
--input_binary=true
