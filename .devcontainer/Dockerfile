# === Base: ROS 2 Humble desktop ===
FROM osrf/ros:humble-desktop-full

# Add ubuntu user with same UID/GID as host
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi

# Add sudo support
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Optional: Webcam support (remove if not needed)
RUN sudo usermod --append --groups video $USERNAME

# Update packages
RUN sudo apt update && sudo apt upgrade -y

########################################################
# Custom installations
########################################################

# Base tools
RUN sudo apt update && sudo apt install -y \
    python3-pip ffmpeg xvfb \
    swig build-essential python3-dev

# Python dependencies (without torch; installed separately with CUDA runtime)
COPY requirements.txt /app/.devcontainer/requirements.txt
RUN pip install --no-cache-dir -r /app/.devcontainer/requirements.txt

# Install PyTorch with bundled CUDA 12.8 runtime
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

########################################################
# End Custom installations
########################################################

# Make sure host NVIDIA driver libs are on the loader path
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/ros/${ROS_DISTRO}/opt/rviz_ogre_vendor/lib:/opt/ros/${ROS_DISTRO}/lib/x86_64-linux-gnu:/opt/ros/${ROS_DISTRO}/lib

# Source ROS setup
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
