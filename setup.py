from setuptools import find_packages, setup
from typing import List

def get_requirements(req_path:str)->List[str]:
    '''
    Fnction to return the list of requirements.
    '''
    with open("requirements.txt", "r") as file:
        requirements = file.read().splitlines()
        try:
            requirements.remove('-e .')
        except:
            pass
        
    return requirements

setup(
    name="assignment_1",
    version="0.0.1",
    author="Apratim",
    author_email='abajpai8@gatech.edu',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)