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
--epochs 20 --schedule 10 25 --gamma 0.1 --wd 1e-4

# TEMP SCALING
python temp_scaling_svhn.py -a resnet --depth 20 \
--resume checkpoints/svhn/14-April-svhn_resnet_depth=20_lossname=NLL_lr=0.1/checkpoint.pth.tar



# TINY-IMAGENET

# TRAIN
python tiny_imagenet.py -a resnet --depth 32 \
--epochs 250 --schedule 100 150 200 --gamma 0.1 --wd 1e-4

# TEMP SCALING



# CIFAR10

# TRAIN:

# RESNET-32
# Define rest of the parameters in the cifar.py file

# TRAIN
python train.py \
--epochs 5 \
--lossname NLL \
--model resnet18 \
--dataset svhn \
--prefix temp-train

python temp_scaling.py \
--model resnet18 \
--dataset svhn \
--resume checkpoints/svhn/resnet18/temp-train-NLL/checkpoint.pth