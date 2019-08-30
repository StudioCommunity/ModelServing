import os
import mimetypes
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms as T


def remove_datauri_prefix(data_uri):
    """Remove prefix of a data URI to base64 content."""
    return data_uri.split(',')[-1]

def img_to_datauri(img):
    """
    convert 
    data : cv2 image Mat
    """
    data = cv2.imencode('.jpg', img)[1].tobytes()
    filetype = "jpg"
    data64 = u''.join(base64.encodebytes(data).decode('ascii').splitlines())
    #cv2.imwrite('outputs/test_3.png',gray)
    return u'data:image/%s;base64,%s' % (filetype, data64)

def pil_img_to_tensor(img):
  to_tensor = T.ToTensor()
  tensor = to_tensor(img)
  return tensor

def tensor_to_imgfile(image_tensor: torch.Tensor, filename):
  # specified for stargan, of which the model accept tensor of 4 dimensions
  image_tensor = image_tensor.squeeze(0)
  to_pil = T.ToPILImage()
  img = to_pil(image_tensor)
  img.save(filename, format="JPEG")

def tensor_to_datauri(image_tensor: torch.Tensor):
  # specified for stargan, of which the model accept tensor of 4 dimensions
  image_tensor = image_tensor.squeeze(0)
  to_pil = T.ToPILImage()
  img = to_pil(image_tensor)
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  data64 = base64.b64encode(buffered.getvalue()).decode("utf8")
  filetype = "jpg"
  return u'data:image/%s;base64,%s' % (filetype, data64)

def imgfile_to_data(filename):
    with open(filename, 'rb') as image:
      image_read = image.read()
    #   image_64_encode = base64.encodebytes(image_read).decode('ascii')
      image_64_encode = u''.join(base64.encodebytes(image_read).decode('ascii').splitlines())
    return image_64_encode

def imgfile_to_datauri(filename):
    """Convert a file (specified by a  filename) into a data URI."""
    if not os.path.exists( filename):
        raise FileNotFoundError
    mime, _ = mimetypes.guess_type( filename)
    with open( filename, 'rb') as fp:
        data = fp.read()
        data64 = u''.join(base64.encodebytes(data).decode('ascii').splitlines())
        return u'data:%s;base64,%s' % (mime, data64)

def base64str_to_ndarray(base64_string):
  base64_string = remove_datauri_prefix(base64_string)
  imgData = base64.b64decode(base64_string)
  nparr = np.fromstring(imgData, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  #print(img.shape)
  #cv2.imwrite("outputs/test_0_origin.png", img)
  return img

def base64str_to_image(base64_string):
  base64_string = remove_datauri_prefix(base64_string)
  return Image.open(BytesIO(base64.b64decode(base64_string)))

def _write_file(str, filename):
  with open(filename, "w") as fp:
    fp.write(str)

# python -m dstest.preprocess.datauri_util
if __name__ == '__main__':
  uri = imgfile_to_datauri("inputs/mnist/sample_0.png")
  #print(uri)
  img = base64str_to_ndarray(remove_datauri_prefix(uri))
  cv2.imwrite("outputs/test1.jpg", img)
  #print(img)
  uri1 = img_to_datauri(img)
  # print(img)
  print(uri1 == uri)
  #_write_file(uri1, "uri1.txt")
  # _write_file(uri, "uri.txt")
  img = base64str_to_ndarray(remove_datauri_prefix(uri1))
  cv2.imwrite("outputs/test.jpg", img)

  uri = imgfile_to_datauri("inputs/stargan/clement.jpg")
  img = base64str_to_image(uri)
  tensor = pil_img_to_tensor(img)
  from . import stargan
  tensor = stargan.transform_image_stargan(img)
  tensor_to_imgfile(tensor, "outputs/clement.jpg")