from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'car_racing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/**/*.py', recursive=True)),
        (os.path.join('share', package_name, 'config'), glob('config/**/*.yaml', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='kevin-eppacher@hotmail.de',
    description='CarRacing app package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'car_racing = car_racing.car_racing_node:main',
            'manual_eval_node = car_racing.manual_eval_node:main',
            'cartpole_test = car_racing.test.cartpole_test:main',
            'car_racing_test = car_racing.test.car_racing_test:main',
            'eval_car_racing = car_racing.test.eval_car_racing:main',
            'car_racing_custom_env = car_racing.test.car_racing_custom_env:main',
        ],
        'rl_tasks': [
            'car_racing = car_racing.spec:CarRacingSpec',
        ],
    },
)
