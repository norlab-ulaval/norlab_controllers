from setuptools import setup

setup(
    name='norlabcontrollib',
    packages=['norlabcontrollib',
              'norlabcontrollib.controllers',
              'norlabcontrollib.path',
              'norlabcontrollib.models',
              'norlabcontrollib.util'],
    version='0.1.0',
    description='Norlab control library',
    author='Dominic Baril',
    license='BSD',
    install_requires=['numpy',
                      'scipy',
                      'pyyaml',
                      'casadi'],
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)
