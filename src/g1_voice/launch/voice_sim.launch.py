from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('g1_voice')
    params_file = os.path.join(pkg_share, 'config', 'voice_params.yaml')

    return LaunchDescription([

        LogInfo(msg='Starting G1 voice pipeline (sim mode)...'),

        # Wake word + SELECT button trigger
        Node(
            package='g1_voice',
            executable='wake_word_node',
            name='wake_word_node',
            output='screen',
            parameters=[params_file],
        ),

        # Speech to text
        Node(
            package='g1_voice',
            executable='stt_node',
            name='stt_node',
            output='screen',
            parameters=[params_file],
        ),

        # LLM — using bob_llm
        # llm_prompt  <- robot_state_node enriches transcript with live state
        # llm_response -> tts_node subscribes here
        # NOTE: keep default 'llm_prompt' topic (do NOT remap to /g1/stt/transcript)
        # so robot_state_node's enrichment is not bypassed.
        Node(
            package='bob_llm',
            executable='llm_node.py',
            name='llm_node',
            output='screen',
            parameters=[params_file],
        ),

        # Text to speech
        Node(
            package='g1_voice',
            executable='tts_node',
            name='tts_node',
            output='screen',
            parameters=[params_file],
        ),

        # Gamepad SELECT → fixed-phrase speech via /g1/voice/say_text.
        # Same wiring as voice_real so SELECT is testable in sim.
        Node(
            package='g1_voice',
            executable='button_trigger_node',
            name='button_trigger_node',
            output='screen',
            parameters=[params_file],
        ),

        # Robot state - enriches transcripts with battery/IMU/network context.
        # Degrades safely to "unknown" when the Unitree state topics are absent
        # (as on a dev machine), matching the real-robot pipeline wiring.
        Node(
            package='g1_voice',
            executable='robot_state_node',
            name='robot_state_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
