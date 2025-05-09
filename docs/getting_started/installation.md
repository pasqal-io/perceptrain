## Requirements

Perceptrain is fully tested on Linux/MacOS operating systems. For Windows users, we recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/about) to install a Linux distribution of choice.

## Installation

Perceptrain can be installed from PyPI with `pip` as follows:

```bash
pip install perceptrain
```

By default, this will also install [PyQTorch](https://github.com/pasqal-io/pyqtorch), a differentiable state vector simulator which serves as the main numerical backend for Perceptrain.

## Install from source

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `perceptrain` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

!!! warning
    `hatch` will not combine nicely with other environment managers such Conda. If you want to use Conda,
    install it from source using `pip`:

    ```bash
    # within the Conda environment
    python -m pip install -e .
    ```

## Citation

If you use perceptrain for a publication, we kindly ask you to cite our work using the following BibTex entry:

```latex
@article{perceptrain2024pasqal,
  title = {perceptrain},
  author={Manu Lahariya},
  year = {2025}
}
```
