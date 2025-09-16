from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cartpole'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/**/*.py', recursive=True)),
        (os.path.join('share', package_name, 'config'), glob('config/**/*.yaml', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='kevin-eppacher@hotmail.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cartpole = cartpole.test.cartpole:main',
            'cartpole_custom_env = cartpole.test.cartpole_custom_env:main',
            'cartpole_vec_env = cartpole.test.cartpole_vec_env:main',
        ],
        'rl_tasks': [
            'cartpole = cartpole.spec:CartpoleSpec',
            'cartpole_vec = cartpole.spec:CartpoleVecSpec',
        ],
    },
)
