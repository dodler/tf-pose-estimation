import argparse

import tfcoreml as tf_converter
import coremltools
from coremltools.models.neural_network.quantization_utils import *

parser = argparse.ArgumentParser(description='Script for graph convertation')

parser.add_argument('--graph', type=str, required=True)
parser.add_argument('--out-name', type=str, required=True)

args = parser.parse_args()

tf_converter.convert(tf_model_path=args.graph,
                     mlmodel_path=args.out_name,
                     output_feature_names=['Openpose/concat_stage7:0'],
                     image_input_names=['image:0'],
                     input_name_shape_dict={'image:0': [1, 224, 224, 3]})

mdl = coremltools.models.MLModel(args.out_name)

q = quantize_weights(mdl, 16, 'linear')
quant_mdl = coremltools.models.MLModel(q).save(args.out_name + '_16b.coreml')
