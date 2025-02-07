#!/bin/bash

# Install Intel Client GPU. Install the Intel graphics GPG public key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Continue installing Client GPU and OpenVINO Runtime based on Ubuntu OS version
OS_VER=$(lsb_release -sr | cut -d'.' -f1)
echo $OS_VER
if [[ $OS_VER -le 24 ]]; then

    # Configure the repositories.intel.com package repository
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
	sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

    # Update the package repository meta-data
    sudo apt update

    # Install the compute-related packages
    apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc

    # Install PyTorch dependencies
    apt-get install -y libze-dev intel-ocloc
    
    # Install OpenVINO Runtime    
    curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64.tgz --output openvino_2024.6.0.tgz
    tar -xf openvino_2024.6.0.tgz
    sudo mv l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64 /opt/intel/openvino_2024.6.0    
    
elif [[ $OS_VER -le 22 ]]; then

    # Configure the repositories.intel.com package repository
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
	sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

    # Update the package repository meta-data
    sudo apt update

    # Install the compute-related packages
    apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo

    # Install PyTorch dependencies    
    apt-get install -y libze-dev intel-ocloc
    
    # Install OpenVINO Runtime
    curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64.tgz --output openvino_2024.6.0.tgz
    tar -xf openvino_2024.6.0.tgz
    sudo mv l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64 /opt/intel/openvino_2024.6.0

else
    echo "Only Ubuntu 24.04 and 22.04 supported. Canceling..."
    exit 1
fi

# Finish installing OpenVINO Runtime
cd /opt/intel/openvino_2024.6.0
sudo -E ./install_dependencies/install_openvino_dependencies.sh

# Install python dependencies
python3 -m pip install -r ./python/requirements.txt

# Create symbolic link
cd /opt/intel
sudo ln -s openvino_2024.6.0 openvino_2024

# Configure enviornment
echo "source /opt/intel/openvino_2024/setupvars.sh" >> ~/.bashrc
source ~/.bashrc

# Done
exit 0
