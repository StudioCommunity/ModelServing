from setuptools import setup

# python setup.py install
setup(
    name="builtin_models",
    version="0.0.1",
    description="builtin_models",
    packages=["builtin_models"],
    install_requires=[
          "cloudpickle",
          "PyYAML"
      ],
    author="Xin Zou",
    license="MIT",
    include_package_data=True,
)