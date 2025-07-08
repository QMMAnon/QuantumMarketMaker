from setuptools import setup, find_packages

setup(
    name="mbt_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gym==0.26.2",
        "numpy",
        "pandas",
        "tensorflow==2.15.0"
    ]
)
