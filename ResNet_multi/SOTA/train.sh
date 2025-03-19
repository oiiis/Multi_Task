# check which model we can run 
python train_compare.py

# Available models:
#  - ImageNet
#  - BEASM-Fully
#  - BEASM-Semi
#  - SRF
#  - SHN
#  - ACNN
#  - U-Net++
#  - OANet 
#  - ResNet10
#  - ResNet18
#  - ResNet34
#  - ResNet50
#  - ResNet101
#  - ResNet152

python train_compare.py ResNet18
python train_compare.py ResNet50
python train_compare.py ResNet101

# finally
python plot.py