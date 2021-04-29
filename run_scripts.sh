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
--epochs 200 \
--schedule-steps 100 150 \
--lossname NLL \
--model resnet32 \
--dataset cifar100 \
--prefix 28-April

python train.py \
--epochs 100 \
--schedule-steps 50 70 \
--lossname LS+MDCA \
--model resnet20 \
--beta 50 --alpha 0.05 \
--dataset svhn \
--prefix 27-April

python train.py \
--epochs 200 \
--schedule-steps 100 150 \
--lossname NLL+MDCA \
--model resnet32 \
--beta 50 \
--dataset cifar100 \
--prefix 27-April

python train.py \
--epochs 200 \
--schedule-steps 50 70 \
--lossname NLL+DCA \
--model resnet32 \
--beta 20 \
--dataset cifar100 \
--prefix 27-April

python temp_scaling.py \
--model resnet20 \
--dataset svhn \
--resume old_checkpoints/svhn/14-April-svhn_resnet_depth=20_lossname=NLL+DCA_beta=1_lr=0.1/checkpoint.pth.tar

python temp_scaling.py \
--model resnet32 \
--dataset cifar100 \
--resume old_checkpoints/svhn/14-April-svhn_resnet_depth=20_lossname=NLL+DCA_beta=1_lr=0.1/checkpoint.pth.tar


python eval.py \
--model resnet34 \
--dataset cifar10 \
--resume checkpoints/cifar10/resnet34/19-April-LSFL_alpha=0.1_gamma=1.0/checkpoint.pth


python dirichilit.py \
--model resnet32 \
--dataset cifar100 \
--epochs 500 \
--lr 0.001 \
--optimizer adam \
--regularizer l2 \
--resume old_checkpoints/cifar100/10-April-cifar100_resnet_depth=32_lossname=NLL_lr=0.1/checkpoint.pth.tar

# To generate logits for post-hoc calibration
python gen_logits.py \
--model resnet32 \
--dataset cifar100 \
--resume old_checkpoints/cifar100/10-April-cifar100_resnet_depth=32_lossname=NLL_lr=0.1/checkpoint.pth.tar

# TO RUN FOR DIRI
python dirichilit.py \
--model resnet32 \
--dataset cifar100 \
--epochs 500 \
--lr 0.001 \
--optimizer adam \
--regularizer l2 \
--resume new NLL model trained

python dirichilit.py \
--model resnet32 \
--dataset cifar100 \
--epochs 500 \
--lr 0.001 \
--optimizer adam \
--regularizer odir \
--resume new NLL model trained