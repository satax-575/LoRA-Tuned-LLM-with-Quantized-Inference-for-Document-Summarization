"""
Package setup for QLoRA-Tuned Llama 3.1 8B Document Summarization.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="qlora-llm-summarization",
    version="1.0.0",
    author="Self Project",
    description="QLoRA-Tuned Llama 3.1 8B for Domain-Adaptive Document Summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "qlora-train=training.train:main",
            "qlora-benchmark=evaluation.benchmark:main",
            "qlora-serve=inference.main:main",
            "qlora-quantize=quantization.awq_quantize:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
