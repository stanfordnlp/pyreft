from setuptools import setup, find_packages

# Read the content of the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the content of the requirements.txt file
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="pyreft",
    version="0.0.2",
    description="REFT: Representation Finetuning for Language Models",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/stanfordnlp/reft",
    author="Zhengxuan Wu",
    author_email="wuzhengx@stanford.edu",
    license="Apache License 2.0",
    packages=find_packages(include=['pyreft', 'pyreft.*']),
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
