#install tf-coreml
#https://github.com/tf-coreml/tf-coreml
#pip install -e .

import tfcoreml as tf_converter
import coremltools

#转coreml
tf_converter.convert(tf_model_path = 'squeezenet_model.pb',
                     mlmodel_path = 'squeezenet_model.mlmodel',
                     input_name_shape_dict = {'input_1:0' : [1, 227, 227, 3]},
                     image_input_names = 'input_1:0',
                     output_feature_names = ['dense_1/BiasAdd:0'], 
                     is_bgr = True,
                    )

#double转float
model_spec = coremltools.utils.load_spec('./squeezenet_model.mlmodel')
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, 'squeezenet_model_16.mlmodel')
