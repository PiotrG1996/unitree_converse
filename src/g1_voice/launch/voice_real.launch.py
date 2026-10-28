from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('g1_voice')
    params_file = os.path.join(pkg_share, 'config', 'voice_params_real.yaml')

    log_level = LaunchConfiguration('log_level', default='INFO')

    return LaunchDescription([

        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='ROS log level (DEBUG/INFO/WARN/ERROR/FATAL) applied to all nodes'),

        LogInfo(msg='Starting G1 voice pipeline (real robot)...'),

        Node(
            package='g1_voice',
            executable='wake_word_node',
            name='wake_word_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
        ),

        Node(
            package='g1_voice',
            executable='stt_node',
            name='stt_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
        ),

        Node(
            package='bob_llm',
            executable='llm_node.py',
            name='llm_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
            # NOTE: llm_node subscribes to its default 'llm_prompt' topic, which is
            # fed by robot_state_node (transcript enriched with live battery/IMU/
            # network state). Do NOT remap llm_prompt -> /g1/stt/transcript here, or
            # the robot_state enrichment is bypassed and the LLM only sees the raw
            # transcript and never gets real robot state.
        ),

        Node(
            package='g1_voice',
            executable='tts_node',
            name='tts_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
        ),
        Node(
            package='g1_voice',
            executable='button_trigger_node',
            name='button_trigger_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
        ),
        Node(
            package='g1_voice',
            executable='robot_state_node',
            name='robot_state_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
        ),
    ])
