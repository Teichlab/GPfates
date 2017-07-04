from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read()



setup(
        name='GPfates',
        version='1.0.0',
        description='Model transcriptional cell fates as mixture of Gaussian Processes',
        long_description=readme(),
        packages=find_packages(),
        install_requires=requirements(),
        author='Valentine Svensson',
        author_email='valentine@nxn.se',
        license='MIT'
    )
