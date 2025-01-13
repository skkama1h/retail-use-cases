#!/bin/bash

# one-time installs
if [ "$1" == "--skip" ]
then
	echo "Skipping chapterization dependencies"
else
	echo "Installing chapterization dependencies"
	sudo apt update
	sudo apt install python3-venv git ffmpeg vim python3 python-is-python3 python3-pip wget -y
	# neo/opencl drivers 24.45.31740.9
	mkdir neo
	cd neo
	wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.1.12/intel-igc-core-2_2.1.12+18087_amd64.deb
	wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.1.12/intel-igc-opencl-2_2.1.12+18087_amd64.deb
	wget https://github.com/intel/compute-runtime/releases/download/24.45.31740.9/intel-level-zero-gpu-dbgsym_1.6.31740.9_amd64.ddeb
	wget https://github.com/intel/compute-runtime/releases/download/24.45.31740.9/intel-level-zero-gpu_1.6.31740.9_amd64.deb
	wget https://github.com/intel/compute-runtime/releases/download/24.45.31740.9/intel-opencl-icd-dbgsym_24.45.31740.9_amd64.ddeb
	wget https://github.com/intel/compute-runtime/releases/download/24.45.31740.9/intel-opencl-icd_24.45.31740.9_amd64.deb
	wget https://github.com/intel/compute-runtime/releases/download/24.45.31740.9/libigdgmm12_22.5.2_amd64.deb
	sudo dpkg -i *.deb
	# sudo apt install ocl-icd-libopencl1
	cd ..

fi
echo "Installing chapterization"
python3 -m venv langchain_chapterization_env
source langchain_chapterization_env/bin/activate

python -m pip install --upgrade pip
pip install wheel setuptools langchain-openai langchain_community
pip install --upgrade-strategy eager "optimum[openvino,nncf]" langchain-huggingface
git clone https://github.com/gsilva2016/langchain.git
pushd langchain; git checkout openvino_asr_loader; popd
pip install -e langchain/libs/community
