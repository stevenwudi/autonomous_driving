import numpy as np
import matplotlib.pyplot as plt

thresholdSetOverlap = [x/float(20) for x in range(21)]
thresholdSetError = range(0, 51)
LINE_COLORS = ['b','g','r','c','m','y','k', '#880015', '#FF7F27', '#00A2E8', '#880015', '#FF7F27', '#00A2E8']


def main():
    valid_dict = np.load(r'C:\Users\steve\Desktop\Figures\tracking_plot\valid.npy')
    test_dict = np.load(r'C:\Users\steve\Desktop\Figures\tracking_plot\test.npy')

    v_dict = valid_dict.item()
    t_dict = test_dict.item()

    v_errCenter = np.vstack(v_dict['errCoverage'])
    v_iou_2d =np.vstack(v_dict['iou_2d'])
    v_errCenter_realworld = v_dict['errCenter_realworld']
    ################################################################
    successRateList = []
    length = v_iou_2d.shape[1]
    for threshold in thresholdSetOverlap:
        seqSuccessList = []
        for seq_iou_2d in v_iou_2d:
            seqSuccess = [score for score in seq_iou_2d if score > threshold]
            seqSuccessList.append(len(seqSuccess)/float(length))
        successRateList.append(seqSuccessList)

    successRateList_plot = [np.mean(x) for x in successRateList]
    ################################################################
    precisionRateList = []
    for threshold in thresholdSetError:
        seqPrecisionRateList = []
        for seq_errCenter in v_errCenter:
            seqPrecisions = [score for score in seq_errCenter if score < threshold]
            seqPrecisionRateList.append(len(seqPrecisions)/float(length))
        precisionRateList.append(seqPrecisionRateList)
    precisionRateList_plot = [np.mean(x) for x in precisionRateList]


    plot_graph_success(successRateList_plot, name='GT_detector')
    plot_graph_precision(precisionRateList_plot, name='GT_detector')
    plt.show()


def plot_graph_success(successRateList_plot, name='GT_detector'):
    plt.figure(num=0, figsize=(9, 6), dpi=70)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ls = '-'
    if len(successRateList_plot) == len(thresholdSetOverlap):
        ave = sum(successRateList_plot) / (len(successRateList_plot))
        plt.plot(thresholdSetOverlap, successRateList_plot, c='r', label='{0} [{1:.3f}]'.format(name, ave), lw=2.0, ls=ls)
    else:
        print('err')
    plt.title('Success plots of {0} (sequence average)'.format(name))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetOverlap[0], thresholdSetOverlap[len(thresholdSetOverlap)-1]+0.1, 0.1))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    # plt.savefig(BENCHMARK_SRC + 'graph/{0}_sq.png'.format(evalType), dpi=74, bbox_inches='tight')
    return plt


def plot_graph_precision(precisionRateList_plot, name='GT_detector'):

    plt.figure(num=1, figsize=(9, 6), dpi=70)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # some don't have precison list--> we will delete them?
    ls = '-'
    ave = sum(precisionRateList_plot) / (len(precisionRateList_plot))
    ave_20 = precisionRateList_plot[20]

    plt.plot(thresholdSetError, precisionRateList_plot, c='r', label='{0} [mean:{1:.3f}, 20p:{2:.3f}]'.format(name, ave, ave_20), lw=2.0, ls=ls)

    plt.title('Precision plots of {0}(sequence average)'.format(name))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetError[0], thresholdSetError[len(thresholdSetError)-1], 10))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    plt.show()
    return plt


if __name__ == '__main__':
    main()
