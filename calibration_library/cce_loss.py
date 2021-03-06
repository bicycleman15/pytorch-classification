import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# @neelabh17 implementation


class CCELossFast(torch.nn.Module):
    def __init__(self, n_classes, n_bins = 10, mode = "eval", loss_type = "sce"):
        '''
        output = [n_Class, h , w] np array: The complete probability vector of an image
        target = [h , w] np array: The GT for the image
        n_bins = [h , w] np array: Number of bins for the Calibration division

        '''
        super(CCELossFast,self).__init__()
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.mode = mode
        self.loss_type = loss_type


        self.createBins()

        self.no_pred_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        self.no_acc_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        self.conf_sum_tot = torch.zeros(self.n_classes, self.n_bins).cuda()

    def reset(self):
        self.no_pred_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        self.no_acc_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        self.conf_sum_tot = torch.zeros(self.n_classes, self.n_bins).cuda()


    def forward(self , output, target):
        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image

        create an three array of [n_class, n_bins]
        -> Number of prediciton array for that specification
        -> Number of correct prediction for that class
        -> Percentge of correct 
        '''

        current_no_pred_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        current_no_acc_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        current_conf_sum_tot = torch.zeros(self.n_classes, self.n_bins).cuda()
        

        output = torch.softmax(output, dim=1)
        # [batch, classes]
        
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            mask = (output> bin_lower) * (output <= bin_upper)

            for class_id in range(self.n_classes):


                class_mask = mask[:,class_id]

                classwise_gt = (target == class_id).long()

                current_no_pred_tot[class_id][i] = torch.sum(class_mask)
                current_no_acc_tot[class_id][i] = torch.sum(class_mask *  classwise_gt)
                current_conf_sum_tot[class_id][i] = torch.sum((output[:,class_id])[class_mask])


        self.no_pred_tot += current_no_pred_tot.data
        self.no_acc_tot += current_no_acc_tot.data
        self.conf_sum_tot += current_conf_sum_tot.data

        avg_acc = (current_no_acc_tot)/(current_no_pred_tot + 1e-13)
        avg_conf = current_conf_sum_tot / (current_no_pred_tot + 1e-13)
        # overall_cceLoss = torch.sum(torch.abs(avg_acc - avg_conf) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))
        # overall_cceLoss = torch.sum(((avg_acc - avg_conf)**2))

        assert (self.loss_type=="sce" or self.loss_type=="kernel" or self.loss_type=="diff")

        # Correct implementation
        if(self.loss_type=="sce"):
            overall_cceLoss = torch.sum(torch.abs(avg_acc - avg_conf) * current_no_pred_tot/torch.sum(current_no_pred_tot))

        # Kernel based implementation
        elif(self.loss_type=="kernel"):
            overall_cceLoss = torch.sum((1-torch.exp((-1*((avg_acc - avg_conf)**2))/0.5)) * (current_no_pred_tot/torch.sum(current_no_pred_tot)))
            
        # difference based approach
        elif(self.loss_type=="diff"):
            overall_cceLoss = torch.sum(((avg_acc - avg_conf)**2))

        # overall_cceLoss = torch.sum(12500*(1-torch.exp((-1*((avg_acc - avg_conf)**2))/6400)) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))
        # print(self.conf_sum_tot.requires_grad)
        return overall_cceLoss

    def createBins(self):

        #uniform bin spacing
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.avg_bin = torch.Tensor((self.bin_lowers + self.bin_uppers)/2).cuda()
        

    def get_perc_table(self, classes):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        self.perc *= 100
        
        from tabulate import tabulate
        x= list(self.perc.cpu().numpy())

        for i in range(len(x)):
            x[i]=list(x[i])
            x[i]=[classes[i]]+list(x[i])
        print(tabulate(x, headers = ["Classes"]+[ "{:0.2f} - {:0.2f}".format(self.bin_lowers[i] * 100, self.bin_uppers[i] * 100) for i in range( len(self.bin_lowers))]))
        
        return self.perc

    def get_diff_score(self):
        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        return torch.sum (torch.abs(avg_acc-avg_conf))/(self.n_bins*self.n_classes)

    def get_overall_CCELoss(self):
        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        # overall_cceLoss = torch.sum(torch.abs(avg_acc - avg_conf) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))
        # overall_cceLoss = torch.sum(((avg_acc - avg_conf)**2))

        # Correct implementation
        # overall_cceLoss = torch.sum(((avg_acc - avg_conf)**2) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))

        # Non Squared  implementation
        overall_cceLoss = torch.sum((torch.abs(avg_acc - avg_conf)) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))

        # Kernel based implementation
        # overall_cceLoss = torch.sum((1-torch.exp((-1*((avg_acc - avg_conf)**2))/0.5)) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))
        # overall_cceLoss = torch.sum(12500*(1-torch.exp((-1*((avg_acc - avg_conf)**2))/6400)) * (self.no_pred_tot/torch.sum(self.no_pred_tot)))

        # print("Overall CCE Loss = ", overall_cceLoss)

        return overall_cceLoss

        
    def get_classVise_CCELoss(self, classes):
        avg_acc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        # print(avg_acc.shape)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        # print(avg_conf.shape)

        x = torch.sum(torch.abs(avg_acc-avg_conf) * self.no_pred_tot, dim = 1) / torch.sum(self.no_pred_tot, dim = 1)
        x = x.reshape(-1,1)

        # print(x.shape)

        x=list(x)
        from tabulate import tabulate
        for i in range(len(x)):
            x[i]=list(x[i])
            x[i]=[classes[i]]+list(x[i])
        print(tabulate(x, headers = ["Classes", "ECELoss"]))

    def get_diff_mean_std (self):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        self.perc *= 100
        avg_conf *= 100
        dif = torch.abs(avg_conf- self.perc)
        return dif.mean(), dif.std()

    def get_perc_table_img(self, classes, tempFileName):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        self.perc *= 100
        avg_conf *= 100


        # Plotting for table
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(self.perc.cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()

        filename = "temp_files/" + tempFileName + ".jpg"
        plt.savefig(filename)
        import cv2
        img_table = cv2.imread(filename)
        # print(img_table.shape)
        
        
        # Plotting for dif map
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(torch.abs(self.perc-avg_conf).cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        plt.savefig(filename)
        plt.clf()

        import cv2
        img_dif = cv2.imread(filename)
        # print(img_dif.shape)
        return img_table, img_dif

    def get_count_table_img(self, classes, tempFileName):
        self.perc = (self.no_acc_tot)/(self.no_pred_tot + 1e-13)
        avg_conf = self.conf_sum_tot / (self.no_pred_tot + 1e-13)
        self.perc *= 100
        avg_conf *= 100


        # Plotting for table
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(100*(self.no_pred_tot/(torch.sum(self.no_pred_tot))).cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        filename = "temp_files/" + tempFileName + ".jpg"
        plt.savefig(filename)
        import cv2
        img_table = cv2.imread(filename)
        # print(img_table.shape)
        
        
        # Plotting for dif map not required in this case
        fig, ax = plt.subplots(figsize=(6,8))
        bin_str_label=[ "{} - {}".format(int(self.bin_lowers[i] * 100), int(self.bin_uppers[i] * 100)) for i in range( len(self.bin_lowers))]
        im, cbar = heatmap(torch.abs(self.perc-avg_conf).cpu().numpy(), classes, bin_str_label, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7)

        fig.tight_layout()
        # plt.show()
        plt.savefig(filename)
        plt.clf()

        import cv2
        img_dif = cv2.imread(filename)
        # print(img_dif.shape)
        return img_table, img_dif

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,fontsize=8)
    ax.set_yticklabels(row_labels,fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
    

