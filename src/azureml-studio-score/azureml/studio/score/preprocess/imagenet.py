import skimage
import skimage.transform

# we may rewrite this logic using torchvision transforms.
def transform_image_imagenet(img, target_size = (224,224)):
  """
  img: ndarray
  """
  img = img * 1.0
  # img = img / 255.0
  # assert (0 <= img).all() and (img <= 1.0).all()
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to targetSize
  resized_img = skimage.transform.resize(crop_img, target_size)

  return resized_img