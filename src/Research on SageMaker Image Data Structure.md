## Abstract

I went through the SageMaker image related documents, and spent some dollar to finish the notebook in their tutorial (most parts are free, but the traing GPU instance is not). I wish what I learned as in this essay is worthwhile. :(

Most concisely, I got two observations.

- SageMaker support RecordIO and ordinary Image Foramt as image data structure, which we can learn from when designing our ImageDirectory.
- SageMaker support standard MIME image/* and non-standard application/x-image as web service input, and application/json as output. Which won't fit in our legacy web servic input/output structure. We may have to use string of base64 encoded image or plain string of image bytes to fill in our json-format input.

Details are as follows.

## Training

SageMake has two modes of training, which support different data types:
- Training in Pipe: Training job streams data directly from Amazon S3
- Training in File: Loads all of data from Amazon S3 to the training instance volumes.

Mode | Support DataTypes
---|---
Training in Pipe | RecordIO, Augmented Manifest Image Format
Training in File | Image Format

- [**RecordIO**](#recordio) is a Apache Mesos binary data format which packs descriptive header (typically contains label information) with the image iwnto bytes.
- [**Image Foramt**](#image-format) is ordinary image folder, must use with another .lst file of format: 
  ~~~
  index \t label_index \t path_to_image
  ~~~
- [**Augmented Manifest Image Format**](#augmented-manifest-image-format) is a json format which contains description of image and its path in S3.

Click the inline links for details.

## Realtime Inference

Realtime inference support image/png, image/jpeg and image-x-image(infer image format from data) as input, and application/json as classification service output. Didn't find doc/example for image generation service, should support the same content types as image input, presumably.

### Input

| ContentType | IsStandard |
| --- | --- |
| image/png | Yes |
| image/jpeg | Yes |
| image/x-image | No |


### Output

| ContentType | IsStandard |
| --- | --- |
| application/json | Yes |

### Example: 

~~~python
import json
import numpy as np

with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
result = response['Body'].read()
# result will be in json format and convert it to ndarray
result = json.loads(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))
~~~

## Batch Inference

Batch Inference support RecordIO or list of jsons (specified with non-standard MIME type application/jsonlines) as input and output. application/x-image	and image/* are also supported but since these can pass only one image per pipeline run, they are not worth digging here.

I haven't find sample code for batch inference, so the detail is to be enriched.

### Input/Output

| ContentType | SplitType |
| --- | --- |
| application/x-recordio-protobuf | RecordIO |
| application/jsonlines | Line |

jsonlines example:
~~~
accept: application/jsonlines
 
 {"prediction": [prob_0, prob_1, prob_2, prob_3, ...]}
 {"prediction": [prob_0, prob_1, prob_2, prob_3, ...]}
 {"prediction": [prob_0, prob_1, prob_2, prob_3, ...]}
~~~

## Train-Deploy Lifecycle Example

[IMAGE CLASSIFICATION WITH RESNET](https://sagemaker-workshop.com/builtin/resnet.html)

## References

### RecordIO
RecordIO is the name for a set of binary data exchange formats. The basic idea is to divide the data into individual chunks, called ‘records’, and then to prepend to every record its length in bytes, followed by the data.
[RecordIO Data Format](http://mesos.apache.org/documentation/latest/recordio/)

For image scenario, RecordIO packs a header and a image into one record.
- Storing images in a compact format--e.g., JPEG, for records--greatly reduces the size of the dataset on the disk. (Lower space consumption)
- Packing data together allows continuous reading on the disk. (Higher performance)
- RecordIO has a simple way to partition, simplifying distributed setting. (Easy to partition)

#### Example
[caltec-256-60-train.rec](s3://xiaoming-debugging/train/caltech-256-60-train.rec)
~~~
import mxnet as mx

record = mx.recordio.MXRecordIO('/Users/minggu/Downloads/caltech-256-60-train.rec', 'r')
item = record.read()
header, img = mx.recordio.unpack_img(item)

print(f"type(item) = {type(item)}")
print(f"header = {header}")
print(f"type(img) = {type(img)}")
import cv2
cv2.imwrite('img.jpg', img)
from IPython.display import Image
Image("img.jpg")
~~~
~~~
type(item) = <class 'bytes'>
header = HEADER(flag=0, label=54.0, id=0, id2=0)
type(img) = <class 'numpy.ndarray'>
~~~
![image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1573921909032&di=d21767efbf321e35926af0c6201f5ecc&imgtype=0&src=http%3A%2F%2Fimg.wdjimg.com%2Fmms%2Ficon%2Fv1%2F1%2F64%2Fe81653d8ba944ae6cfe49ad429a20641_256_256.png)

### Image Format
S3 bucket + .lst file, have to be "manually" splitted if needed
~~~
image_index  label_index  image_path
5      1   your_image_directory/train_img_dog1.jpg
1000   0   your_image_directory/train_img_cat1.jpg
22     1   your_image_directory/train_img_dog2.jpg
~~~

### Augmented Manifest Image Format
The manifest file format should be in JSON Lines format in which each line represents one sample. The images are specified using the 'source-ref' tag that points to the S3 location of the image.
~~~
{"source-ref":"s3://image/filename1.jpg", "class":"0"} 
{"source-ref":"s3://image/filename2.jpg", "class":"1", "class-metadata": {"class-name": "cat", "type" : "groundtruth/image-classification"}}
~~~

## Links
1. [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)
2. [Image Classification with ResNet](https://sagemaker-workshop.com/builtin/resnet.html)
3. [RecordIO Data Format](http://mesos.apache.org/documentation/latest/recordio/)
4. [Common Data Formats for Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html)
5. [Create a Dataset Using RecordIO](https://mxnet.apache.org/api/faq/recordio)
6. [Caltech256 Dataset Intro](http://www.vision.caltech.edu/Image_Datasets/Caltech256/intro/)
7. [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
8. [Azure Open Dataset MNIST](https://azure.microsoft.com/en-us/services/open-datasets/catalog/mnist/)