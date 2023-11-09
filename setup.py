from setuptools import setup, find_packages

setup(
    name='AA4HMRL',
    version='0.1.0',
    packages=find_packages(include=['modules/*', 'src/*', 'models/', 'ccnn/*'])
)