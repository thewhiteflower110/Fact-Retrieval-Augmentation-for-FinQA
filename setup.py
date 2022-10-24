from setuptools import setup,find_packages

setup(
    name='Fact-Retrieval-Augmentation-for-FinQA',
    version='0.1.0',
    packages=find_packages(include=['Fact-Retrieval-Augmentation-for-FinQA', 'Fact-Retrieval-Augmentation-for-FinQA.*']),
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'jupyter',
        'tokenizers==0.12.1',
        'torch==1.11.0',
        'torchmetrics==0.9.2',
        'torchvision==0.12.0',
        'tqdm==4.64.0',
        'transformers>=4.20.1',
        'huggingface-hub==0.8.1'
    ]
)
