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

# Create symbolic link
cd /opt/intel
sudo ln -s openvino_2024.6.0 openvino_2024

# Configure enviornment
echo "source /opt/intel/openvino_2024/setupvars.sh" >> ~/.bashrc
source ~/.bashrc

# Install Conda
source activate-conda.sh

# one-time installs
if [ "$1" == "--skip" ]
then
	echo "Skipping qna dependencies"
	activate_conda
else
	echo "Installing qna dependencies"
	sudo apt update
	sudo apt install -y curl git ffmpeg vim portaudio19-dev build-essential wget -y

	CUR_DIR=`pwd`
        cd /tmp
	miniforge_script=Miniforge3-$(uname)-$(uname -m).sh
	[ -e $miniforge_script ] && rm $miniforge_script
	wget "https://github.com/conda-forge/miniforge/releases/latest/download/$miniforge_script"
	bash $miniforge_script -b -u
	# used to activate conda install
	activate_conda
	conda init
	cd $CUR_DIR

	# neo/opencl drivers 24.45.31740.9
	mkdir neo
	cd neo
	wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb
	wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb
	wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb
	wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb
	wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb
	wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb
	wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb
	sudo dpkg -i *.deb
	# sudo apt install ocl-icd-libopencl1
	cd ..

	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html --create-dirs -o ./docs/ovms_what_is_openvino_model_server.html
	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_metrics.html -o ./docs/ovms_docs_metrics.html
	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_streaming_endpoints.html -o ./docs/ovms_docs_streaming_endpoints.html
	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_target_devices.html -o ./docs/ovms_docs_target_devices.html
fi

# Create python enviornment
conda create -n ovlangvidsumm python=3.10 -y
conda activate ovlangvidsumm
pip install -r requirements.txt

# Done
exit 0
