 # Setup

 - Poetry: https://github.com/sdispater/poetry (`curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`)
 - pre-commit: https://pre-commit.com/ (`pip install pre-commit`)
 - setup python environment:
    1. make sure python3.7 is active
    2. run in root folder: `poetry install`
 - using pycharm: mark `src` as sources root directory
- if running tensorboard raises an error about setuptools, run: `poetry run pip install -U setuptools`
