from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="langchain-agentic-ai",
    version="2.0.0",
    author="AI Developer",
    author_email="developer@example.com",
    description="Modern LangChain-based Agentic AI System with Multi-Tool Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/langchain-agentic-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "langchain-agent=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "langchain",
        "ai",
        "agent",
        "llm",
        "artificial-intelligence",
        "machine-learning",
        "nlp",
        "rag",
        "vector-search",
        "multi-tool",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/langchain-agentic-ai/issues",
        "Source": "https://github.com/yourusername/langchain-agentic-ai",
        "Documentation": "https://github.com/yourusername/langchain-agentic-ai#readme",
    },
)
