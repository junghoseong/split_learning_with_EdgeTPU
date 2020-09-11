"""
Pretrain되어 있는 MobileNetV2를 keras model으로서 불러오고, 이를 .h5 file으로 저장한다.
그 후, 해당 keras model에 integer-only inference를 위한 post-training quantization을 적용하고, 이를 .tflite file으로 저장한다. 
"""
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import argparse

resize_image_first_call = True

def resize_image(image, desired_size, pad_color=0):
	h, w = image.shape[:2]
	desired_h, desired_w = desired_size, desired_size

	# interpolation method
	if h > desired_h or w > desired_w: # shrinking image
	    interp = cv2.INTER_AREA
	else: # stretching image
	    interp = cv2.INTER_CUBIC

	# aspect ratio of image
	aspect = float(w) / h 
	desired_aspect = float(desired_w) / desired_h

	if (desired_aspect > aspect) or ((desired_aspect == 1) and (aspect <= 1)):  # new horizontal image
	    new_h = desired_h
	    new_w = np.round(new_h * aspect).astype(int)
	    pad_horz = float(desired_w - new_w) / 2
	    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
	    pad_top, pad_bot = 0, 0

	elif (desired_aspect < aspect) or ((desired_aspect == 1) and (aspect >= 1)):  # new vertical image
	    new_w = desired_w
	    new_h = np.round(float(new_w) / aspect).astype(int)
	    pad_vert = float(desired_h - new_h) / 2
	    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
	    pad_left, pad_right = 0, 0

	# set pad color
	if len(image.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
	    pad_color = [pad_color]*3

	# scale and pad
	scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interp)
	scaled_image = cv2.copyMakeBorder(scaled_image, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

	return scaled_image

def representative_images_gen(imageset_dir, num_samples, desired_size):
	"""
	imageset_dir에서 num_samples만큼의 image data를 sample하여 preprocess한 뒤 
	이들의 list를 representative_images로 출력한다.
	"""
	representative_images = []
	image_paths = os.listdir(imageset_dir)
	for i in range(num_samples):
		image_path = image_paths[i]
		image = cv2.imread(os.path.join(imageset_dir, image_path))
		resized_image = resize_image(image, desired_size)
		normalized_image = resized_image.astype(np.float32) / 255.0
		representative_image = np.reshape(normalized_image, (1,desired_size,desired_size,3))
		representative_images.append(representative_image)
		if i == 0 or (i + 1) % 1000 == 0:
			print('%d번째 representative image processing 완료'%(i+1))
	return representative_images

def split(full_model, split_boundary):
	layer_names = [layer.name for layer in full_model.layers]
	split_boundary_idx = layer_names.index(split_boundary) 

	A_input = tf.keras.Input(tensor=full_model.input)
	x = A_input
	for layer in full_model.layers[1:split_boundary_idx]:
		if layer.name == 'block_2_add':
			x = layer([x, full_model.layers[layer_names.index('block_1_project_BN')].output])
		elif layer.name == 'block_4_add':
			x = layer([x, full_model.layers[layer_names.index('block_3_project_BN')].output])
		elif layer.name == 'block_5_add':
			x = layer([x, full_model.layers[layer_names.index('block_4_add')].output])
		elif layer.name == 'block_7_add':
			x = layer([x, full_model.layers[layer_names.index('block_6_project_BN')].output])
		elif layer.name == 'block_8_add':
			x = layer([x, full_model.layers[layer_names.index('block_7_add')].output])
		elif layer.name == 'block_9_add':
			x = layer([x, full_model.layers[layer_names.index('block_8_add')].output])
		elif layer.name == 'block_11_add':
			x = layer([x, full_model.layers[layer_names.index('block_10_project_BN')].output])
		elif layer.name == 'block_12_add':
			x = layer([x, full_model.layers[layer_names.index('block_11_add')].output])
		elif layer.name == 'block_14_add':
			x = layer([x, full_model.layers[layer_names.index('block_13_project_BN')].output])	
		elif layer.name == 'block_15_add':
			x = layer([x, full_model.layers[layer_names.index('block_14_add')].output])
		else:
			x = layer(x)
	full_A_model = tf.keras.Model(inputs=A_input, outputs=x)
	for idx in range(1, len(full_A_model.layers)):
		full_A_model.layers[idx].set_weights(full_model.layers[idx].get_weights())
	full_A_model.summary()

	B_input = tf.keras.Input(tensor=full_model.layers[split_boundary_idx].input)
	x = B_input
	for layer in full_model.layers[split_boundary_idx:]:
		if layer.name == 'block_2_add':
			x = layer([x, full_model.layers[layer_names.index('block_1_project_BN')].output])
		elif layer.name == 'block_4_add':
			x = layer([x, full_model.layers[layer_names.index('block_3_project_BN')].output])
		elif layer.name == 'block_5_add':
			x = layer([x, full_model.layers[layer_names.index('block_4_add')].output])
		elif layer.name == 'block_7_add':
			x = layer([x, full_model.layers[layer_names.index('block_6_project_BN')].output])
		elif layer.name == 'block_8_add':
			x = layer([x, full_model.layers[layer_names.index('block_7_add')].output])
		elif layer.name == 'block_9_add':
			x = layer([x, full_model.layers[layer_names.index('block_8_add')].output])
		elif layer.name == 'block_11_add':
			x = layer([x, full_model.layers[layer_names.index('block_10_project_BN')].output])
		elif layer.name == 'block_12_add':
			x = layer([x, full_model.layers[layer_names.index('block_11_add')].output])
		elif layer.name == 'block_14_add':
			x = layer([x, full_model.layers[layer_names.index('block_13_project_BN')].output])	
		elif layer.name == 'block_15_add':
			x = layer([x, full_model.layers[layer_names.index('block_14_add')].output])
		else:
			x = layer(x)
	full_B_model = tf.keras.Model(inputs=B_input, outputs=x)
	for idx in range(1, len(full_B_model.layers)):
		full_B_model.layers[idx].set_weights(full_model.layers[len(full_A_model.layers) - 1 + idx].get_weights())
	full_B_model.summary()

	with open('./output/full_model_weight.txt', 'w') as f:
		for item in full_model.get_weights():
			f.write('%s\n'%item)	
	with open('./output/full_A_model_weight.txt', 'w') as f:
		for item in full_A_model.get_weights():
			f.write('%s\n'%item)
	with open('./output/full_B_model_weight.txt', 'w') as f:
		for item in full_B_model.get_weights():
			f.write('%s\n'%item)

	return full_A_model, full_B_model, layer_names, split_boundary_idx

def main():
	# Argument parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('--split_boundary', required=True, help='The name of split boundary layer.')
	parser.add_argument('--imageset_dir', required=True, help='Path to an imageset for representative image(s)')
	parser.add_argument('--num_samples', default=5000, help='Number of samples in a representative imageset')
	parser.add_argument('--desired_size', default=224, help='Desired input image size. In specific, an image is resized to (desired_size) X (desired_size)')
	parser.add_argument('--output_dir', default='./output', help='Path to save outputs')
	args = parser.parse_args()
	if not os.path.exists(args.imageset_dir):
		sys.exit('%s does not exist!' % args.imageset_dir)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Pretrained MobileNetV2를 불러오고 그 개요를 출력
	# MobileNetV2의 classifier 부분은 불러오지 않는다. (i.e., include_top=False)
	full_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, pooling='avg')
	print('Successfully imported full model')
	full_model.summary()

	full_A_model, full_B_model, layer_names, split_boundary_idx = split(full_model, args.split_boundary)
	print('Successfully split full model into part A and B')

	# full_A_model과 full_B_model을 각각 .h5 file으로 저장
	MobileNetV2_with_imagenet_full_A_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full_A.h5') 
	MobileNetV2_with_imagenet_full_A_pb_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full_A_pb') 
	MobileNetV2_with_imagenet_full_B_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full_B.h5') 
	
	full_A_model.save(MobileNetV2_with_imagenet_full_A_pb_path, save_format='tf')
	imported = tf.saved_model.load(MobileNetV2_with_imagenet_full_A_pb_path)

	tf.keras.models.save_model(full_A_model, MobileNetV2_with_imagenet_full_A_path)
	tf.keras.models.save_model(full_B_model, MobileNetV2_with_imagenet_full_B_path)
	print('Successfully saved part A to {}'.format(MobileNetV2_with_imagenet_full_A_path))
	print('Successfully saved part B to {}'.format(MobileNetV2_with_imagenet_full_B_path))

	# 각 model에 integer-only inference를 위한 post-training quantization을 적용
	representative_images = representative_images_gen(args.imageset_dir, args.num_samples, args.desired_size)
	def representative_data_gen_A():
		for representative_image in representative_images:
			yield [representative_image]
	def representative_data_gen_B():
		for representative_image in representative_images:
			x = representative_image
			"""
			for layer in full_model.layers[1:split_boundary_idx]:
				if layer.name == 'block_2_add':
					x = layer([x, full_model.layers[layer_names.index('block_1_project_BN')].output])
				elif layer.name == 'block_4_add':
					x = layer([x, full_model.layers[layer_names.index('block_3_project_BN')].output])
				elif layer.name == 'block_5_add':
					x = layer([x, full_model.layers[layer_names.index('block_4_add')].output])
				elif layer.name == 'block_7_add':
					x = layer([x, full_model.layers[layer_names.index('block_6_project_BN')].output])
				elif layer.name == 'block_8_add':
					x = layer([x, full_model.layers[layer_names.index('block_7_add')].output])
				elif layer.name == 'block_9_add':
					x = layer([x, full_model.layers[layer_names.index('block_8_add')].output])
				elif layer.name == 'block_11_add':
					x = layer([x, full_model.layers[layer_names.index('block_10_project_BN')].output])
				elif layer.name == 'block_12_add':
					x = layer([x, full_model.layers[layer_names.index('block_11_add')].output])
				elif layer.name == 'block_14_add':
					x = layer([x, full_model.layers[layer_names.index('block_13_project_BN')].output])	
				elif layer.name == 'block_15_add':
					x = layer([x, full_model.layers[layer_names.index('block_14_add')].output])
				else:
					x = layer(x)
			"""
			x = imported(x)
			representative_feature = x.numpy()
			yield [representative_feature]

	converter_A = tf.lite.TFLiteConverter.from_keras_model(full_A_model)
	converter_A.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_A.representative_dataset = representative_data_gen_A
	converter_A.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter_A.inference_input_type = tf.uint8
	converter_A.inference_output_type = tf.uint8
	quant_A_model = converter_A.convert()
	print('Successfully generated quantized part A')

	converter_B = tf.lite.TFLiteConverter.from_keras_model(full_B_model)
	converter_B.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_B.representative_dataset = representative_data_gen_B
	converter_B.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter_B.inference_input_type = tf.uint8
	converter_B.inference_output_type = tf.uint8
	quant_B_model = converter_B.convert()
	print('Successfully generated quantized part B')

	# 각 Quantized model을 .tflite file으로 저장
	MobileNetV2_quant_A_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_quant_A.tflite')
	MobileNetV2_quant_B_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_quant_B.tflite')
	open(MobileNetV2_quant_A_path, 'wb').write(quant_A_model)	
	print('Successfully saved the quantized part A to {}'.format(MobileNetV2_quant_A_path))
	open(MobileNetV2_quant_B_path, 'wb').write(quant_B_model)	
	print('Successfully saved the quantized part B to {}'.format(MobileNetV2_quant_B_path))

	# 사전에 edgetpu-compiler를 install하여야 함
	os.system('edgetpu_compiler {0}'.format(MobileNetV2_quant_A_path))
	os.system('edgetpu_compiler {0}'.format(MobileNetV2_quant_B_path))
	print('Successfully compiled the quantized parts for inference with edge TPU')

if __name__ == '__main__':
  main()