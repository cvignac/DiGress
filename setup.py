from setuptools import setup, find_packages

setup(
    name='DiGress',
    version='1.0.0',
    url=None,
    author='A.S., C.V.',
    author_email='author@gmail.com',
    description='Discrete denoising diffusion for graph generation',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1']
)