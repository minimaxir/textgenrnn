from setuptools import setup, find_packages

long_description = '''
Easily train your own text-generating neural network of
any size and complexity on any text dataset with a few lines
of code, or quickly train on a text using a pretrained model.

- A modern neural network architecture which utilizes new techniques as
attention-weighting and skip-embedding to accelerate training
and improve model quality.
- Able to train on and generate text at either the
character-level or word-level.
- Able to configure RNN size, the number of RNN layers,
and whether to use bidirectional RNNs.
- Able to train on any generic input text file, including large files.
- Able to train models on a GPU and then use them with a CPU.
- Able to utilize a powerful CuDNN implementation of RNNs
when trained on the GPU, which massively speeds up training time as
opposed to normal LSTM implementations.
- Able to train the model using contextual labels,
allowing it to learn faster and produce better results in some cases.
- Able to generate text interactively for customized stories.
'''


setup(
    name='textgenrnn',
    packages=['textgenrnn'],  # this must be the same as the name above
    version='2.0.0',
    description='Easily train your own text-generating neural network ' \
    'of any size and complexity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Max Woolf',
    author_email='max@minimaxir.com',
    url='https://github.com/minimaxir/textgenrnn',
    keywords=['deep learning', 'tensorflow', 'keras', 'text generation'],
    classifiers=[],
    license='MIT',
    python_requires='>=3.5',
    include_package_data=True,
    install_requires=['h5py', 'scikit-learn', 'tqdm', 'tensorflow>=2.1.0']
)
