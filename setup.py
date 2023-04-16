from setuptools import find_packages
from setuptools import setup


CLASSIFIERS = ['License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']


URL = 'https://github.com/brandontrabucco/mass'


KEYWORDS = ['Deep Learning', 'Neural Networks',
            'Reinforcement Learning', 'Visual Room Rearrangement']


with open('README.md', 'r') as README:
    setup(name='mass', version='1.0.0', license='MIT',
          packages=find_packages(include=['mass', 'mass.*']),
          description='A Simple Approach For Visual Room Rearrangement: 3D Mapping & Semantic Search (ICLR 2023)',
          classifiers=CLASSIFIERS, long_description=README.read(),
          long_description_content_type='text/markdown',
          author='Brandon Trabucco', author_email='brandon@btrabucco.com',
          url=URL, download_url=URL + '/archive/v1_0_0.tar.gz',
          keywords=KEYWORDS, install_requires=["torch"])
