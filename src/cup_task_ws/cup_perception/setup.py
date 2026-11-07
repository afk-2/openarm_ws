from glob import glob

from setuptools import find_packages, setup

package_name = 'cup_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='afk',
    maintainer_email='afk@todo.todo',
    description='Cup pose sources for the cup task (Gazebo ground truth now, vision later)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gazebo_cup_pose = cup_perception.gazebo_cup_pose:main',
            'cup_pose_estimator = cup_perception.cup_pose_estimator:main',
            'd435_depth_noise = cup_perception.d435_depth_noise:main',
            'evaluate_perception = cup_perception.evaluate_perception:main',
        ],
    },
)
