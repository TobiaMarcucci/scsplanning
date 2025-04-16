from setuptools import setup

setup(
    name='scsplanning',
    version='0.1.0',
    description='A biconvex method for designing smooth trajectories that traverse sequences of convex sets in minimum time.',
    url='https://github.com/TobiaMarcucci/scsplanning',
    author='Tobia Marcucci',
    author_email='tobiam@mit.edu',
    license='MIT',
    packages=['scsplanning'],
    install_requires=['numpy', 'pybezier'],
)
