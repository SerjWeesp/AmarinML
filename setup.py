from setuptools import setup, find_packages

setup(
    name='AmarinML',  # Name of your package
    version='0.1',  # Initial version
    packages=find_packages(),  # Automatically find the sub-packages in your directory
    install_requires=[],  # List any dependencies here, e.g. ['numpy', 'pandas']
    description='A package containing useful ML functions',  # Short description of your package
    author='Sergey Amarin',
    author_email='serj.amarin@gmail.com',
    url='https://github.com/SerjWeesp/AmarinML',
)