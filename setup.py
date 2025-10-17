from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read()


def get_version():
    with open('GPfates/__init__.py', 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split('=')[1].strip(' \'"')


setup(
        name='GPfates',
        version=get_version(),
        description='Model transcriptional cell fates as mixture of Gaussian Processes',
        long_description=readme(),
        packages=find_packages(),
        install_requires=requirements(),
        author='Valentine Svensson',
        author_email='valentine@nxn.se',
        license='MIT'
    )
