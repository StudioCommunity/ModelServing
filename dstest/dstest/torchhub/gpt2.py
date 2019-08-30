#pip install tqdm boto3 requests regex
import torch

# python -m dstest.torchhub.gpt2
if __name__ == '__main__':
  ### First, tokenize the input
  #############################
  tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2Tokenizer', 'gpt2')

  #  Prepare tokenized input
  text_1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
  text_2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
  tokenized_text_1 = tokenizer.tokenize(text_1)
  tokenized_text_2 = tokenizer.tokenize(text_2)
  indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
  indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
  tokens_tensor_1 = torch.tensor([indexed_tokens1])
  tokens_tensor_2 = torch.tensor([indexed_tokens2])


  ### Get the hidden states computed by `gpt2Model`
  #################################################
  model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2Model', 'gpt2')
  model.eval()

  # Predict hidden states features for each layer
  # past can be used to reuse precomputed hidden state in a subsequent predictions
  with torch.no_grad():
    hidden_states_1, past = model(tokens_tensor_1)
    hidden_states_2, past = model(tokens_tensor_2, past=past)


  ### Predict the next token using `gpt2LMHeadModel`
  ##################################################
  lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2LMHeadModel', 'gpt2')
  lm_model.eval()

  # Predict hidden states features for each layer
  with torch.no_grad():
    predictions_1, past = lm_model(tokens_tensor_1)
    predictions_2, past = lm_model(tokens_tensor_2, past=past)

  # Get the predicted last token
  predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
  predicted_token = tokenizer.decode([predicted_index])
  print(predicted_token)
  assert predicted_token == ' who'


  ### Language modeling and multiple choice classification `gpt2DoubleHeadsModel`
  ###############################################################################
  double_head_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2DoubleHeadsModel', 'gpt2')
  double_head_model.eval() # Set the model to train mode if used for training

  tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
  mc_token_ids = torch.LongTensor([[len(tokenized_text_1) - 1, len(tokenized_text_2) - 1]])

  with torch.no_grad():
      lm_logits, multiple_choice_logits, presents = double_head_model(tokens_tensor, mc_token_ids)

  print(lm_logits)
  print(multiple_choice_logits)
  print(presents)