import skimage
import skimage.transform
import numpy as np
import pandas as pd
import tensorflow as tf
from . import imagenet
from dstest.postprocess import prob_to_category
from builtin_score import tensorflow_score_module
from builtin_score.tensorflow_score_module import TensorflowScoreModule

def load_image(path, target_size = (224,224)):
  img = skimage.io.imread(path)
  transformed = imagenet.transform_image_imagenet(img, target_size)
  return transformed

# convert this to a generic postprocess module
synset = [l.strip() for l in open('model/vgg/synset.txt').readlines()]

model_path = "model/vgg/"

def load_model(sess):
  # VGG-16: https://github.com/ry/tensorflow-vgg16
  # \\clement-pc1\share\clwan\model\vgg16-model files
  return tensorflow_score_module.load_graph("model/vgg/vgg16-20160129.tfmodel", sess)

def load_inputs():
  imgs=[]
  for i in ['cat.jpg', "dog1.jpg", "dog2.jpg"]:
    img = load_image('inputs/imagenet/'+i)
    skimage.io.imsave(f"outputs/{i}-transform.jpg", img)
    img = img / 255.0
    img = img.flatten()
    imgs.append(img)
  df = pd.DataFrame()
  df.insert(len(df.columns), "import/images", imgs, True)
  return df

# python -m dstest.preprocess.loadvgg
if __name__ == '__main__':
  df = load_inputs()
  print(df)
  
  import yaml
  with open(model_path + "model_spec.yml") as fp:
      config = yaml.safe_load(fp)

  tfmodule = TensorflowScoreModule(model_path, config)
  result_df = tfmodule.run(df)
  print(result_df)
  prob = result_df["import/prob"]

  result = prob_to_category.get_categories(prob, synset)
  print ("The category is :",result)
