# CIFAR10

# TRAIN:

# RESNET-32
# Define rest of the parameters in the cifar.py file

# TRAIN
python cifar10.py -a resnet --depth 32 \
--epochs 200 --schedule 100 150 --gamma 0.1 --wd 1e-4

# TEMP SCALING
python temp_scaling.py -a resnet --depth 32 \
--resume checkpoints/cifar10/9-April-cifar10_resnet_depth=32_lossname=NLL_lr=0.1/checkpoint.pth.tar



# CIFAR100

# TRAIN
python cifar100.py -a resnet --depth 32 \
--epochs 200 --schedule 100 150 --gamma 0.1 --wd 1e-4

# TEMP SCALING
python temp_scaling_cifar100.py -a resnet --depth 32 \
--resume checkpoints/cifar100/10-April-cifar100_resnet_depth=32_lossname=LS+MDCA_beta=40_alpha=0.1_lr=0.1/checkpoint.pth.tar


# SVHN

# TRAIN
python svhn.py -a resnet --depth 20 \
--epochs 35 --schedule 18 28 --gamma 0.1 --wd 1e-4

# TEMP SCALING