from setuptools import setup, find_packages

setup(
    name="motionVectorDeinterlacing",
    version="1.0.0",
    description="Real-Time Motion Vector Deinterlacing and Super-Resolution",
    author="Vince",
    packages=find_packages(), # 会自动找到 motionVectorDeinterlacing 文件夹
    python_requires=">=3.8",
)