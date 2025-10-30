from setuptools import setup, find_packages

setup(
    name="aqi-forecast",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
)