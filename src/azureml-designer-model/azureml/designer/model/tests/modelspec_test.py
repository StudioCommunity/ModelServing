import pytest

def test_success():
    assert 4 == 4

@pytest.mark.skip(reason="TODO: remove hardcoded model path")
def test_keras_model():
    from azureml.designer.modelspec.keras import save_model
    print('---keras test---')
    from keras.models import load_model
    model = load_model('D:/enlistment/aml/CustomModules/dstest/model/keras-mnist/model.h5')
    print('------')
    save_model(model, "./tmp/outputModels/keras/")
    print('********')

if __name__ == '__main__':
    # # keras test
    # from azureml.designer.modelspec.keras import *
    # print('---keras test---')
    # from keras.models import load_model
    # model = load_model('D:/GIT/CustomModules-migu-NewYamlTest2/dstest/model/keras-mnist/model.h5')
    # print('------')
    # save_model(model, "./test/outputModels/keras/")
    # print('********')

    # #sklearn test
    # from azureml.designer.modelspec.sklearn import *
    # print('---sklearn test---')
    # import pickle
    # with open('D:/GIT/CustomModules-migu-NewYamlTest2/dstest/dstest/sklearn/model/sklearn/model.pkl', "rb") as f:
    #     model = pickle.load(f)
    # print('------')
    # save_model(model, "./test/outputModels/sklearn/")
    # print('********')

    # #pytorch test
    # from azureml.designer.modelspec.pytorch import *
    # import cloudpickle
    # print('---pytorch test---')
    # with open('D:/GIT/CustomModules-migu-NewYamlTest2/dstest/model/pytorch-mnist/model.pkl', 'rb') as fp:
    #     model = cloudpickle.load(fp)
    # print('------')
    # save_model(model, "./test/outputModels/pytorch/")
    # print('********')
    
    pass