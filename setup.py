from setuptools import setup, find_packages, Extension

#https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/setup.py

setup(
    name='od_toolbox',
    version='0.0.1',
    description='Object detection toolbox',
    packages=find_packages(),
    ext_package='od_toolbox',
    install_requires=[
    ],
)