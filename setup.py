from setuptools import find_packages, setup

def get_requirements(file_path: str):
    requirements = []
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
    
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name = 'End-To-End-Ml-Project',
    version = '0.0.1',
    author = 'Abdullah',
    author_email = 'abdullahrizwan354@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)
