from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simple_yolo'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages = find_packages(include=['simple_yolo', 'simple_yolo.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include model files
        (f'share/{package_name}/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools','ultralytics'],
    zip_safe=True,
    maintainer='mj',
    maintainer_email='eric.mjkim35@gmail.com',
    description='YOLO integrated package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera = simple_yolo.camera:main',
            'perception_node = simple_yolo.perception_node:main',
            'perception_client = simple_yolo.perception_client:main',
        ],
    },
)

