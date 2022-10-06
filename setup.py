from setuptools import find_packages, setup

setup(
    name='norlabcontrollib',
    # packages=find_packages(include=['norlabcontrollib'),
    packages=['norlabcontrollib', 'norlabcontrollib.controllers', 'norlabcontrollib.path'],
    version='0.1.0',
    description='Norlab control library',
    author='Dominic Baril',
    license='BSD',
    install_requires=['numpy', 'scipy', 'pyyaml'],
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)