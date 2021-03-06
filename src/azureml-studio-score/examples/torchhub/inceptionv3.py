import torch

# python -m dstest.torchhub.inceptionv3
if __name__ == '__main__':
  model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)
  model.eval()

  # Download an example image from the pytorch website
  import urllib
  url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
  try: urllib.URLopener().retrieve(url, filename)
  except: urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
  from PIL import Image
  from torchvision import transforms
  input_image = Image.open(filename)
  preprocess = transforms.Compose([
      transforms.Resize(299),
      transforms.CenterCrop(299),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
  print("input_batch: ", input_batch.shape)

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
    output = model(input_batch)

  # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
  print("output: ", output.shape)
  #print(output[0])
  # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
  prob = torch.nn.functional.softmax(output[0], dim=0)
  print(prob.shape)
  predicted_index = torch.argmax(prob).item()
  print(predicted_index)