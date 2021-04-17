from .loss import MDCA_LabelSmoothLoss, DCATrainLoss_alpha, MDCA_NLLLoss, FocalLoss, CrossEntropyWrapper, LS_Wrapper

loss_dict = {
    "LS+MDCA" : MDCA_LabelSmoothLoss,
    "FL" : FocalLoss,
    "NLL+MDCA" : MDCA_NLLLoss,
    "NLL" : CrossEntropyWrapper,
    "NLL+DCA" : DCATrainLoss_alpha,
    "LS" : LS_Wrapper
}