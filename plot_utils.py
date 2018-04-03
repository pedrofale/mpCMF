import matplotlib.pyplot as plt

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
    n_boxes = len(low_sep)
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    # first boxplot pair
    bp1 = plt.boxplot(low_sep, positions = range(1, n_boxes + 1), widths = 0.6)
    setBoxColors(bp1, colors)

    # second boxplot pair
    bp2 = plt.boxplot(high_sep, positions = range(n_boxes + 3, n_boxes + 3 + n_boxes), widths = 0.6)
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
    ax.set_xticks([2, 7])

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if title is not None:
        plt.title(title)

    if not show_legend and legend is not None:
        return ax, handles
    
    return ax
