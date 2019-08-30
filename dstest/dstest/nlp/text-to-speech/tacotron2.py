#pip install numpy scipy librosa unidecode inflect librosa
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
import torch
import builtin_models.python
from builtin_score.builtin_score_module import *
from io import BytesIO

def tensor_to_wav(audio):
    buffered = BytesIO()
    rate = 22050
    write(buffered, rate, audio)
    import base64
    data64 = base64.b64encode(buffered.getvalue()).decode("utf8")
    filetype = "wav"
    #write(f"audio_{index}.wav", rate, audio)
    return u'data:audio/%s;base64,%s' % (filetype, data64)

class Tacotron2Model(builtin_models.python.PythonModel):
    def __init__(self, model_path = None):
        self.device = 'cuda' # cuda only
        
        print('# device:', self.device)
        tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
        tacotron2 = tacotron2.to(self.device)
        tacotron2.eval()
        print('# tacotron2 parameters:', sum(param.numel() for param in tacotron2.parameters()))

        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(self.device)
        waveglow.eval()
        print('# waveglow parameters:', sum(param.numel() for param in waveglow.parameters()))
        
        self.tacotron2 = tacotron2
        self.waveglow = waveglow
    
    def predict(self, text):
        # prep-rocessing
        results = []
        for line in text:
            print(f"generate for line: {line}")
            sequence = np.array(self.tacotron2.text_to_sequence(line, ['english_cleaners']))[None, :]
            
            # run the models
            sequence = torch.from_numpy(sequence).to(device=self.device, dtype=torch.int64)
            with torch.no_grad():
                _, mel, _, _ = self.tacotron2.infer(sequence)
                audio = self.waveglow.infer(mel)
                audio_numpy = audio.data.cpu().numpy() # [1, 256000]
                wav = tensor_to_wav(audio_numpy[0])
                results.append(wav)
        return results

# python -m dstest.nlp.text-to-speech.tacotron2
if __name__ == '__main__':
    model_path = "model/tacotron2"
    github = 'StudioCommunity/CustomModules:master'
    module = 'dstest/dstest/nlp/text-to-speech/tacotron2.py'
    model_class = 'Tacotron2Model'

    #model = Tacotron2Model()
    #builtin_models.python.save_model(model, model_path, github = github, module_path = module, model_class= model_class)

    model1 = builtin_models.python.load_model(model_path, github = github, module_path = module, model_class= model_class, force_reload= True)

    text = "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
    x = np.array([text])
    # run the models
    audios = model1.predict(x)
    #print(audios.shape)

    
    d = {'text': x}
    df = pd.DataFrame(data=d)
    # test_tensor(model_path, df)
    module = BuiltinScoreModule(model_path, {"Append score columns to output": "True"})
    result = module.run(df)
    print(result.columns)
    print(f"result: {result}")
    result.to_csv("~/data.csv")
