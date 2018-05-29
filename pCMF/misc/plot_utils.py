import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
import operator

def setBoxColors(bp, colors):
    n_boxes = len(bp['boxes'])
    assert n_boxes <= len(colors)
    
    for i in range(n_boxes):
        plt.setp(bp['boxes'][i], color=colors[i])
        plt.setp(bp['caps'][i*2], color=colors[i])
        plt.setp(bp['caps'][i*2+1], color=colors[i])
        plt.setp(bp['whiskers'][i*2], color=colors[i])
        plt.setp(bp['whiskers'][i*2+1], color=colors[i])
        plt.setp(bp['fliers'][i], color=colors[i])
        plt.setp(bp['medians'][i], color=colors[i])

def plot_simulation_results(low_sep, high_sep, ax=None, legend=None, title=None, ylabel=None,
                            colors = ['blue', 'red', 'green', 'yellow', 'cyan'], show_legend=False):
    # low_sep and high_sep must both be lists of length equal to the number of methods to test (num of boxes)
    # and for each method the array must be of shape nruns

    # if n_boxes is odd, each xticklabel should be at the center box
    # if it is even, each xticklabel should be between the 2 center boxes

    n_boxes = len(low_sep)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    # first boxplot pair
    bp1_pos = range(1, n_boxes + 1)
    bp1 = plt.boxplot(low_sep, positions = bp1_pos, widths = 0.6)
    setBoxColors(bp1, colors)

    # second boxplot pair
    bp2_pos = range(n_boxes + 3, n_boxes + 3 + n_boxes)
    bp2 = plt.boxplot(high_sep, positions = bp2_pos, widths = 0.6)
    setBoxColors(bp2, colors)

    if legend is not None:
        handles = []
        for i in range(n_boxes):
            # draw temporary lines and use them to create a legend
            h, = plt.plot([0,0], colors[i])
            handles.append(h)
        
        if show_legend:
            plt.legend(handles, legend)
            _ = [h.set_visible(False) for h in handles]
    
    # set axes limits and labels
    if show_legend:
        plt.xlim(0,n_boxes*2 + 5)
    else:
        plt.xlim(0, n_boxes*2 + 3)
    ax.set_xticklabels(['Low separability', 'High separability'])
    ax.set_xticks([np.mean(bp1_pos), np.mean(bp2_pos)])

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and legend is not None:
        return ax, handles
    
    return ax    

def plot_convergence_curves(curve_list, label_list=None, ax=None, legend=None, title='', xlabel='', ylabel='', filename=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if label_list is not None:
        assert len(curve_list) == len(label_list)
        for i in range(len(curve_list)):
            plt.plot(curve_list[i], label=label_list[i])
    else:
        for i in range(len(curve_list)):
            plt.plot(curve_list[i])
            
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if legend:
        plt.legend()
    if ax is None:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

def plot_model_convergence(model_list, mode='ll_time', ax=None, legend=None, title='', xlabel='', ylabel='', filename=None):
    """Takes a list of models as input.
    """
    if mode not in ['ll_time', 'll_it', 'silh_time', 'silh_it']:
        return

    if mode == 'll_time':
        curve_list = [model.inf.ll_time for model in model_list]
    if mode == 'll_it':
        curve_list = [model.inf.ll_it for model in model_list]
    if mode == 'silh_time':
        curve_list = [model.inf.silh_time for model in model_list]
    if mode == 'silh_it':
        curve_list = [model.inf.silh_it for model in model_list]

    plot_convergence_curves(curve_list, label_list=[model.name for model in model_list], 
        ax=ax, legend=legend, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename)

def plot_tsne(tsne, clusters, labels=None, ax=None, legend=None, title='', xlabel='', ylabel='', s=30, alpha=0.7, filename=None):
    if labels is not None:
        labels = np.array(labels)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()
    
    C = np.unique(clusters).size
    for c in range(C):
        if labels is not None:
            ax.scatter(tsne[clusters==c, 0], tsne[clusters==c, 1], s=s, alpha=alpha, label=labels[clusters==c][0])
        else:
            ax.scatter(tsne[clusters==c, 0], tsne[clusters==c, 1], s=s, alpha=alpha)
    
    if labels is not None and legend:
        ax.legend()
    
    plt.title(title)
    if ax is None:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

def plot_sorted_tsnes(model_list, clusters, labels=None, ax=None, legend=None, title='', s=30, alpha=0.7, bbox_to_anchor=[1., 1.], filename=None):
    # Sort by decreasing silhouette
    names = []
    tsnes = []
    silhs = []
    for model in model_list:
        names.append(model.name)
        tsnes.append(model.proj_2d)
        silhs.append(model.silhouette)
    scores = dict(zip(names, silhs))
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    
    # Plot in decreasing silhouette order
    if ax is None:
        fig = plt.figure(figsize=(16, 4))

    for i in range(len(model_list)):
        ax = plt.subplot(1, len(model_list), i+1)
        plot_tsne(tsnes[names.index(sorted_scores[i][0])], clusters, labels=labels, ax=ax, s=s, alpha=alpha)
        plt.title(sorted_scores[i][0])
    if labels is not None and legend:
        plt.legend(bbox_to_anchor=bbox_to_anchor, frameon=True)
    
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def plot_imputation_density(imputed, true, dropout_idx, title="", ymax=10, nbins=50, ax=None, cmap="Greys"):
    # imputed is NxP 
    # true is NxP
    
    # We only care about the entries affected by dropouts
    x, y = imputed[dropout_idx], true[dropout_idx]
    
    # let's only look at the values that are lower than ymax
    mask = x < ymax
    x = x[mask]
    y = y[mask]
    
    mask = y < ymax
    x = x[mask]
    y = y[mask]
    
    # make the vectors the same size
    l = np.minimum(x.shape[0], y.shape[0])
    x = x[:l]
    y = y[:l]
    
    data = np.vstack([x, y])

    if ax is None:
        plt.figure(figsize=(5, 5))
    
    axes = plt.gca()
    axes.set_xlim([0, ymax])
    axes.set_ylim([0, ymax])

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:ymax:nbins*1j, 0:ymax:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.title(title, fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel('Original counts')

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap=cmap)

    a, _, _, _ = np.linalg.lstsq(y[:,np.newaxis], x, rcond=None)
    l = np.linspace(0, ymax)
    plt.plot(l, a * l, color='black')

    plt.plot(l, l, color='black', linestyle=":")

def plot_sorted_imputation_densities(model_list, X_train, ax=None, ymax=10, nbins=50, cmap="Greys", filename=None):
    # Sort by decreasing imputation error
    names = []
    dropimp_errs = []
    est_Rs = []
    for model in model_list:
        names.append(model.name)
        est_Rs.append(model.est_R)
        dropimp_errs.append(model.dropimp_err)
    scores = dict(zip(names, dropimp_errs))
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False) # lower is better

    # Plot in decreasing silhouette order
    if ax is None:
        fig = plt.figure(figsize=(20, 4))

    for i in range(len(model_list)):
        ax = plt.subplot(1, len(model_list), i+1)
        plot_imputation_density(est_Rs[names.index(sorted_scores[i][0])], 
            X_train, model_list[i].dropout_idx, ymax=ymax, ax=ax, title=sorted_scores[i][0], nbins=nbins, cmap=cmap)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
