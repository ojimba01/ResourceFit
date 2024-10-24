from setuptools import setup, find_packages

setup(
    name="resourcefit",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        'psutil>=5.9.5',
        'pandas>=2.0.0',
        'requests>=2.31.0',
        'click>=8.0.0',
        'docker>=6.0.0',
    ],
    entry_points={
        'console_scripts': [
            'resourcefit=resourcefit.main:analyze_and_recommend',
        ],
    },
    author="Olayinka Jimba",
    author_email="ojimba01@gmail.com",
    description="A CLI tool to analyze Docker containers and recommend AWS EC2 instances.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ojimba01/resourcefit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
