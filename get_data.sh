mkdir ./save
mkdir ./log
mkdir ./data/pretrained
wget -q ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy -O "./data/pretrained/vgg16.npy"
wget -q http://miniplaces.csail.mit.edu/data/data.tar.gz -O "./data/data.tar.gz"
tar -xzf ./data/data.tar.gz -C ./data
rm -f "./data/data.tar.gz"