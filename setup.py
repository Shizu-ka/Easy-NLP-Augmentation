from setuptools import setup, find_packages

setup(
    name='easy_nlp_augmentation',
    version='1.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'requests',
        'pandas',
        'nlpaug'
    ],
    entry_points={
        'console_scripts': [
            'text_augmenter = text_augmenter.augment:main'
        ]
    },
    author='Shizuka',
    author_email='shizuka0@proton.me',
    description='A package for augmenting text data using NLP techniques directly in your pandas dataframe.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Shizu-ka/easy-nlp-augmentation',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
