from setuptools import setup, find_packages

setup(
    name='textgenrnn',
    packages=['textgenrnn'],  # this must be the same as the name above
    version='1.0',
    description='Pretrained character-based neural network for ' \
    'easily generating text.',
    author='Max Woolf',
    author_email='max@minimaxir.com',
    url='https://github.com/minimaxir/textgenrnn',
    keywords=['deep learning', 'tensorflow', 'keras', 'text generation'],
    classifiers=[],
    license='MIT',
    include_package_data=True,
    install_requires=['keras>=2.1.5', 'h5py', 'scikit-learn']
)
