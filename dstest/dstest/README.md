python -m dstest.preprocess.import_image  --input_path inputs/mnist --output_path datas/mnist
python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --image_column=image --target_column=x --target_datauri_column=x.data --target_image_size=28x28
python -m dstest.tensorflow.mnist_test


# stargan
python -m dstest.preprocess.import_image  --input_path inputs/stargan --output_path datas/stargan
python -m dstest.preprocess.preprocess  --input_path datas/stargan --output_path outputs/stargan --image_column=image --target_column=x --target_datauri_column=x.data --target_image_size=256x256
python -m dstest.pytorch.stargan
python -m dstest.postprocess.tensor_to_image --input_path outputs/stargan/model_output --output_path outputs/stargan/tensor_to_image --tensor_column=0