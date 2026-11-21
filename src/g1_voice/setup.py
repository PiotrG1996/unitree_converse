from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'g1_voice'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Piotr Gapski',
    maintainer_email='maintainer@example.com',  # TODO: public e-mail
    description='Unitree G1 conversation: Unitree GPT online, offline Aletta pipeline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stt_node = g1_voice.stt_node:main',
            'tts_node = g1_voice.tts_node:main',
            'wake_word_node = g1_voice.wake_word_node:main',
            'button_trigger_node = g1_voice.button_trigger_node:main',
            'robot_state_node = g1_voice.robot_state_node:main',
            'assistant_mode_node = g1_voice.assistant_mode_node:main',
        ],
    },
)
