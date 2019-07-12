import torch
from torch.autograd import Variable
import cloudpickle


class PytorchScoreModule(object):

    def __init__(self, model_path):
        with open(model_path, "rb") as fp:
            self.model = cloudpickle.load(fp)
        print(f"Successfully loaded model from {model_path}")

    def run(self, df):
        # Iter over row and predict on a single entry to avoid overflow of GPU Memory
        output_label = []
        for _, row in df.iterrows():
            input_params = row["text_id"] # This is ugly hard coded. Should be specified in model_spec
            x = Variable(torch.LongTensor([input_params]))
            if torch.cuda.is_available():
                x = x.cuda()
            output = self.model(x)
            _, predicted = torch.max(output, 1)
            output_label.append(predicted.view(1).cpu().numpy()[0])

        return output_label

