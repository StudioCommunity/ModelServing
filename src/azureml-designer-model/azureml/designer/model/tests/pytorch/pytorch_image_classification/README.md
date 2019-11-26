
# Command line

- pip install keyring artifacts-keyring

- pip install azureml-designer-model==0.1.20191125.8 --extra-index-url=https://msdata.pkgs.visualstudio.com/_packaging/azureml-modules%40Local/pypi/simple/

- git clone https://msdata.visualstudio.com/DefaultCollection/AzureML/_git/CustomModules

- cd CustomModules

- git checkout migu/ForDensenetTest

- cd src\azureml-designer-model\azureml\designer\model\tests\pytorch\pytorch_image_classification

- python -m training_module.test_densenet_entry
