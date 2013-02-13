# coding=utf-8
import copper
import numpy as np
import matplotlib.pyplot as plt

def histogram(series, bins=20, legend=True, retList=False):
    '''
    Draws a histogram for the selected column on matplotlib

    Parameters
    ----------
        bins: int, number of bins of the histogram, default 20
        legend: boolean, True if want to display the legend of the ploting
        ret_list: boolean, True if want the method to return a list with the
                            distribution(information) of each bin

    Return
    ------
        nothing, figure is ready to be shown
    '''
    plt.hold(True)
    nas = len(series) - len(series.dropna())
    series = series.dropna()
    series = series[series != float('-inf')]
    series = series[series != float('inf')]

    if series.dtype == object:
        types = copper.transform.category_labels(series)
        series = copper.transform.category2number(series)
        bins = len(set(series))

        count, divis = np.histogram(series.values, bins=bins)
        width = 0.97 * (divis[1] - divis[0])

        types = types.tolist()
        types.insert(0, 'NA')
        count = count.tolist()
        count.insert(0, nas)

        labels = ['%s: %d' % (typ, cnt) for cnt, typ in zip(count, types)]
        centers = np.array(range(len(types))) - 0.5

        plt.bar(-width, nas, width=width, color='r', label=labels[0])
        for c, h, t in zip(centers[1:], count[1:], labels[1:]):
            plt.bar(c, h, align = 'center', width=width, label=t)

        plt.xticks(centers, types)
    else:
        bins = bins if len(set(series)) > bins else len(set(series))
        count, divis = np.histogram(series.values, bins=bins)
        width = 0.97 * (divis[1] - divis[0])
        centers = (divis[:-1] + divis[1:]) / 2
        labels = ['%.1f - %.2f: %s' % (i, f, c) for c, i, f in
                                        zip(count, divis[:-1], divis[1:])]
        plt.bar(min(divis) - width, nas, width=width, color='r', label="NA: %d" % nas)
        for c, h, t in zip(centers, count, labels):
            plt.bar(c, h, align = 'center', width=width, label=t)

    if legend:
        plt.legend(loc='best')

    if retList:
        return pd.Series(labels)

def scatter(self, var1, var2, var3=None, **args):
    x = self.frame[var1].values
    y = self.frame[var2].values
    if var3 is None:
        plt.scatter(x, y, **args)
    else:
        z = self.frame[var3].values
        plt.scatter(x, y, c=z, **args)
        plt.gray()
    plt.xlabel(var1)
    plt.ylabel(var2)
