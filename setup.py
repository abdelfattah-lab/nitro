from setuptools import setup, find_packages

setup(
    name='nitro',
    version='0.1',
    description='Serving LLMs on Intel NPUs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anthony Fei',
    author_email='ayf7@cornell.edu',
    url='https://github.com/abdelfattah-lab/nitro',
    packages=find_packages(),
    install_requires=[
        'numpy < 1.27',
        'openvino >= 2024.3.0',
        'openvino-tokenizers >= 2024.3.0.0',
        'transformers >= 4.45',
        'nncf'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)