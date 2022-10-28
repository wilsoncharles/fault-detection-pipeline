from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return a list of requirements
    """ 
    requirement_list:List[str] =[]
    
    """
    Write a code to read requirements.txt and append each requirement to the output list.
    """
    return requirement_list

setup(
    name='sensor',
    version='0.0.1',
    author='wilson',
    author_email= 'wilsoncharles21@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements() ## ["pymango==4.2.0"],  
)