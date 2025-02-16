from setuptools import setup, find_packages

setup(
    name="ASL_detect",
    version="0.1",
    packages=find_packages(),
    package_data={
        'ASL_detect': ['*.pth'],
    },
    install_requires=[],
    author="1603",
    description="A simple library that uses the CNN model we created and makes it easier to use.",
    url="https://gitlab.com/yuv76/asl_detect.git",
)
