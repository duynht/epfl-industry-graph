sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update  
sudo apt-get upgrade  
sudo apt-get install build-essential cmake g++ gfortran git pkg-config python-dev software-properties-common wget
sudo rm -rf /var/lib/apt/lists/*

# Install docker
sudo apt-get remove docker docker-engine docker.io containerd runc
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world

sudo apt-get autoremove 

# # NVIDIA drivers
# sudo add-apt-repository ppa:graphics-drivers/ppa
# sudo apt-get install nvidia-352

# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# sudo systemctl restart docker

# # NVIDIA container toolkit
# sudo apt-get install -y nvidia-container-toolkit

# # Restart
# sudo shutdown -r no
mkdir data && mkdir data/raw && mkdir data/extracted && mkdir data/extracted/patent && mkdir data/extracted/indeed
cd data/raw/
mkdir patent \
    && wget https://drive.switch.ch/index.php/s/kc7YY33wTeukfsJ/download -O patent.zip \
    && unzip patent.zip -d patent/ \
    && rm patent.zip
mkdir indeed \
    && wget https://drive.switch.ch/index.php/s/z4ZvVJy95Yj5AWq/download -O indeed.zip \
    && unzip indeed.zip -d indeed/ \
    && rm indeed.zip
cd ../../

mkdir data/truth \
    && wget https://drive.switch.ch/index.php/s/m2sPKsRJO3KEO0x/download -O related_entities.zip \
    && unzip related_entities.zip -d data/truth/ \
    && rm related_entities

docker build --tag industry-graph:0.1 .
docker run --rm --squash -it -v $PWD:./ industry-graph