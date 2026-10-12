#!/usr/bin/env bash
# source_ros.sh — set up the ROS 2 + CycloneDDS environment for the G1 voice
# pipeline so manual `ros2 topic pub/echo` commands work without OOM-killing.
#
# WHY THIS EXISTS (cline fix 2026-10-10):
#   Manual `ros2 topic pub/echo` commands were dying with
#       bad_alloc caught: std::bad_alloc   ->   Killed
#   The OOM killer showed `ros2` / `_ros2_daemon` ballooning to 9-12 GB RSS.
#   Root cause: the interactive shell exported CYCLONEDDS_URI with an *unexpanded*
#   "~" (e.g. CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml). CycloneDDS cannot
#   parse that path, so it either fell back to its aggressive built-in default
#   config (discovering the entire robot DDS graph across all interfaces ->
#   multi-GB memory -> bad_alloc) or hard-failed with
#       "can't open configuration file ~/cyclonedds_ws/cyclonedds.xml".
#   This script mirrors the EXACT environment the working systemd service uses
#   (absolute CYCLONEDDS_URI + unitree_ros2 sourced), verified stable at ~64 MB
#   peak RSS.
#
# USAGE:
#   source /home/unitree/unitree_converse/source_ros.sh
#   ros2 topic pub --once /g1/stt/transcript std_msgs/msg/String "data: 'Hello'"
#   ros2 topic echo /g1/robot_state
#   ros2 topic echo --once /g1/voice/tts/speaking
#
# NOTE: use `source`, do not execute — it sets env vars in the current shell.

# Guard against double-sourcing harmlessly.
if [ -n "$__UNITREE_CONVERSE_ROS_SOURCED" ]; then
    return 0 2>/dev/null || exit 0
fi
export __UNITREE_CONVERSE_ROS_SOURCED=1

source /opt/ros/foxy/setup.bash

if [ -f "$HOME/cyclonedds_ws/install/setup.bash" ]; then
    source "$HOME/cyclonedds_ws/install/setup.bash"
fi
if [ -f "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash" ]; then
    source "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash"
fi

# Absolute path is REQUIRED — never use a literal "~" here.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="$HOME/cyclonedds_ws/cyclonedds.xml"
export ROS_DOMAIN_ID=0

# The voice pipeline workspace (g1_voice, bob_llm).
if [ -f "$HOME/unitree_converse/install/setup.bash" ]; then
    source "$HOME/unitree_converse/install/setup.bash"
fi

echo "[source_ros.sh] ROS2 env ready: RMW=$RMW_IMPLEMENTATION DOMAIN=$ROS_DOMAIN_ID CYCLONEDDS_URI=$CYCLONEDDS_URI"
