"""Voice pipeline (real): wake word, STT, robot state, LLM, TTS, pad buttons, assistant mode."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NODES = [
    ('g1_voice', 'wake_word_node'),
    ('g1_voice', 'stt_node'),
    ('g1_voice', 'robot_state_node'),
    ('bob_llm', 'llm_node.py'),
    ('g1_voice', 'tts_node'),
    ('g1_voice', 'button_trigger_node'),
    ('g1_voice', 'assistant_mode_node'),
]


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('g1_voice'), 'config', 'voice_params_real.yaml')
    log_level = LaunchConfiguration('log_level')
    return LaunchDescription(
        [DeclareLaunchArgument('log_level', default_value='INFO')] + [
            Node(package=package, executable=executable,
                 name=executable.replace('.py', ''), output='screen',
                 parameters=[params],
                 arguments=['--ros-args', '--log-level', log_level])
            for package, executable in NODES])
