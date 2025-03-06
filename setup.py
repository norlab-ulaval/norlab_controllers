from setuptools import setup

# Function to read requirements.txt
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()
    
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
    install_requires=read_requirements(),
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)