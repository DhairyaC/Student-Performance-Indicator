from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of packages/libraries required for this project
    '''
    HYPHEN_E_DOT = '-e .'
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements


setup(
    name='Student Performance Indicator',
    version='0.0.1',
    author='Dhairya Jayesh Chheda',
    author_email='dhchhed@iu.edu',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)