<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/extras/assets/logo/perceptrain_logo_white" width="75%">
    <source media="(prefers-color-scheme: light)" srcset="./docs/extras/assets/logo/perceptrain_logo.svg" width="75%">
    <img alt="Perceptrain logo" src="./docs/assets/logo/perceptrain_logo.svg" width="75%">
  </picture>
</p>

**Perceptrain** is a Python package that provides a simple interface to execute distributed machine learning training. It supports customization, gradient-based, gradient-free optimizations and various experiment tracking methods.

**For more detailed information, [check out the documentation](https://pasqal-io.github.io/perceptrain/latest/).

**For any questions or comments, [feel free to start a discussion](https://github.com/pasqal-io/perceptrain/discussions).
**

[![Linting](https://github.com/pasqal-io/perceptrain/actions/workflows/lint.yml/badge.svg)](https://github.com/pasqal-io/perceptrain/actions/workflows/lint.yml)
[![Tests](https://github.com/pasqal-io/perceptrain/actions/workflows/test_fast.yml/badge.svg)](https://github.com/pasqal-io/perceptrain/actions/workflows/test_fast.yml)
[![Documentation](https://github.com/pasqal-io/perceptrain/actions/workflows/build_docs.yml/badge.svg)](https://pasqal-io.github.io/perceptrain/latest)
[![Pypi](https://badge.fury.io/py/perceptrain.svg)](https://pypi.org/project/perceptrain/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Coverage](https://img.shields.io/codecov/c/github/pasqal-io/perceptrain?style=flat-square)


## Feature highlights

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/extras/assets/perceptrain_arch.png" width="75%">
    <source media="(prefers-color-scheme: light)" srcset="./docs/extras/assets/perceptrain_arch.png" width="75%">
    <img alt="Perceptrain architecture" src="./docs/assets/perceptrain_arch.png" width="75%">
  </picture>
<p align="center">

* Training models made simple with **Trainer** and **Train Configurations**.

* Support for **gradient based** and **gradient free** optimization.

* **Accelerator** supported distributed training made simple across multi node/multi gpu setups.

* Extensive **callbacks**, along with support for mlflow and tensorboard tracking.

## Installation guide

perceptrain is available on [PyPI](https://pypi.org/project/perceptrain/) and can be installed using `pip` as follows:

```bash
pip install perceptrain
```

The default, pre-installed backend for perceptrain is [PyQTorch](https://github.com/pasqal-io/pyqtorch), a differentiable state vector simulator. It is possible to install additional following extras:

* `mlflow`: For experiment tracking.

To install individual extras, use the following syntax (**IMPORTANT** Make sure to use quotes):

```bash
pip install "perceptrain[mlflow]"
```

To install all available extras, simply do:

```bash
pip install "perceptrain[all]"
```

## Contributing

Before making a contribution, please review our [code of conduct](docs/getting_started/CODE_OF_CONDUCT.md).

- **Submitting Issues:** To submit bug reports or feature requests, please use our [issue tracker](https://github.com/pasqal-io/perceptrain/issues).
- **Developing in perceptrain:** To learn more about how to develop within `perceptrain`, please refer to [contributing guidelines](docs/getting_started/CONTRIBUTING.md).

### Setting up perceptrain in development mode

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `perceptrain` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

**WARNING**
`hatch` will not combine nicely with other environment managers such as Conda. If you still want to use Conda,
install it from source using `pip`:

```bash
# within the Conda environment
python -m pip install -e .
```

Users also report problems running Hatch on Windows, we suggest using WSL2.

## Citation

If you use perceptrain for a publication, we kindly ask you to cite our work using the following BibTex entry:

```latex
@article{perceptrain2024pasqal,
  title = {perceptrain},
  author={Manu Lahariya},
  year = {2025}
}
```

## License
Perceptrain is a free and open source software package, released under the Apache License, Version 2.0.
