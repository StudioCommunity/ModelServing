import pytest


def test_model_factory():
    from azureml.studio.model.model_factory import FlavorRegistry, ModelFactory
    assert len(FlavorRegistry.supported_flavors()) == 0

    FlavorRegistry.get_flavor("pytorch", "cloudpickle")
    assert len(FlavorRegistry.supported_flavors()) > 0

    print(FlavorRegistry.supported_flavors("pytorch"))

    assert FlavorRegistry.get_flavor("invalid") is None
    flavor = {
        "name": "pytorch",
        "serialization_method": "cloudpickle"
    }
    assert ModelFactory.get_model_class(flavor) is not None


if __name__ == '__main__':
    pass
