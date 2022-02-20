from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = HERE.joinpath("README.md").read_text()


def _read_requirements():
    reqs = HERE.joinpath("requirements", "requirements.in")
    with reqs.open("r") as f:
        requirements = [v for line in f.readlines() if (v := line.strip())]
    return [r for r in requirements if not (r.startswith("#") or r.startswith("-") or r.startswith("jax"))]


setup(
    # Metadata
    name="acni",
    author="Miller Wilt",
    author_email="miller@pyriteai.com",
    use_scm_version=True,
    description="A Jax/Optax implementation of Anticorrelated Noise Injection for Improved Generalization",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/PyriteAI/anticorrelated-noise-injection",
    keywords=["artificial intelligence", "machine learning", "jax", "optax", "generalization", "noise injection"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    # Options
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=_read_requirements(),
    python_requires=">=3.8.0",
)
