from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

packages = find_packages(exclude=['contrib', 'docs', 'tests'])
print("packages:")

params = {
    'name': "mdn",
    'version': "0.0.1",
    "long_description": long_description,
    'packages': packages,
    'python_requires': ">=3.0"
}

setup(**params)
