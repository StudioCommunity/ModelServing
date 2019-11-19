from torchvision import transforms as T


def transform_image_stargan(img, resize=(300, 300), crop_size=(256, 256), target_size=(256, 256)):
  """
  img: PIL.JpegImagePlugin.JpegImageFile
  """
  normalize = T.Normalize(
  mean=[0.5, 0.5, 0.5],
  std=[0.5, 0.5, 0.5]
  )

  # need fix later, just to get stargan work
  preprocess = T.Compose([
      T.Resize(resize[0]),
      T.CenterCrop(crop_size[0]),
      T.Resize(target_size[0]),
      T.ToTensor(),
      normalize
      ])
  
  image_tensor = preprocess(img)
  return image_tensor.unsqueeze(0)