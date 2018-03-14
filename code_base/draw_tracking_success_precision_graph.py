import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

thresholdSetOverlap = [x/float(20) for x in range(21)]
thresholdSetError = range(0, 51)
thresholdSetError_RW = [x/float(51) for x in range(51)]
LINE_COLORS = ['b','g','c','m','y','k', '#880015', '#FF7F27', '#00A2E8', '#880015', '#FF7F27', '#00A2E8']

font = {'size': 20}
import matplotlib
matplotlib.rc('font', **font)

def main():
    result_dir ='/media/samsumg_1tb/cvpr_DTA_Results'
    with open(os.path.join(result_dir, 'SEG-LSTM_Shuffle/test-2.pickle'), 'rb') as f:
        SEG_LSTM_dict = pickle.load(f)

    with open(os.path.join(result_dir, 'FrameToSequence_Shuffle/test-2.pickle'), 'rb') as f:
        FrameToSequence_dict = pickle.load(f)

    with open(os.path.join(result_dir, 'FrameToFrame_Shuffle/test-2.pickle'), 'rb') as f:
        FrameToFrame_dict = pickle.load(f)

    scoreList = {}
    length = 8
    ###################### 3D plot ###################
    ErrCenter_RW_List = {'FtF': np.vstack(FrameToFrame_dict['errCenter_realworld']),
                      'FtS':np.vstack(FrameToSequence_dict['errCenter_realworld']), 'SEG-LSTM': np.vstack(SEG_LSTM_dict['errCenter_realworld'])}
    success_Rate_RW_List = {'FtF': np.vstack(FrameToFrame_dict['iou_3d']),
                      'FtS':np.vstack(FrameToSequence_dict['iou_3d']), 'SEG-LSTM': np.vstack(SEG_LSTM_dict['iou_3d'])}
    for seq_name in [ 'FtF', 'FtS', 'SEG-LSTM']:
        successRateList = []
        for threshold in thresholdSetOverlap:
            seqSuccessList = []
            for seq_iou_2d in success_Rate_RW_List[seq_name]:
                seqSuccess = [score for score in seq_iou_2d if score > threshold]
                seqSuccessList.append(len(seqSuccess) / float(length))
            successRateList.append(seqSuccessList)
        scoreList[seq_name] = {}
        scoreList[seq_name]['successRateList'] = [np.mean(x) for x in successRateList]
        scoreList[seq_name]['AUC'] = np.sum(scoreList[seq_name]['successRateList']) / len(thresholdSetOverlap)
    title = 'success_3d'
    plot_graph_success(scoreList, title)

    ################################################################
    threshold_cm = 20
    for seq_name in ['FtF', 'FtS', 'SEG-LSTM']:
        precisionRateList = []
        for threshold in thresholdSetError_RW:
            seqPrecisionRateList = []
            for seq_errCenter in ErrCenter_RW_List[seq_name]:
                seqPrecisions = [score for score in seq_errCenter if score < threshold]
                seqPrecisionRateList.append(len(seqPrecisions) / float(length))
            precisionRateList.append(seqPrecisionRateList)
        precisionRateList_plot = [np.mean(x) for x in precisionRateList]
        scoreList[seq_name] = {}
        scoreList[seq_name]['precisionRateList'] = [np.mean(x) for x in precisionRateList_plot]
        scoreList[seq_name]['p20'] = scoreList[seq_name]['precisionRateList'][threshold_cm]
    title = 'precision_3d'
    plot_graph_precision(scoreList, title, threshold_cm)
    plt.show()

    plt.close('all')
    ###################### 2D plot ###################
    result_dir = '/media/samsumg_1tb/cvpr_DTA_Results/kalman_filter'
    Kalman_test_errCenter = np.load(os.path.join(result_dir, 'Kalman_valid_errCenter.npy'))
    Kalman_test_iou_2d = np.load(os.path.join(result_dir, 'Kalman_valid_iou_2d.npy'))
    ################################################################
    ErrCenter_List = {'Kalman Filter':Kalman_test_errCenter, 'FtF': np.vstack(FrameToFrame_dict['errCoverage']),
                      'FtS':np.vstack(FrameToSequence_dict['errCoverage']), 'SEG-LSTM': np.vstack(SEG_LSTM_dict['errCoverage'])}
    success_Rate_List = {'Kalman Filter':Kalman_test_iou_2d,  'FtF': np.vstack(FrameToFrame_dict['iou_2d']),
                      'FtS':np.vstack(FrameToSequence_dict['iou_2d']), 'SEG-LSTM': np.vstack(SEG_LSTM_dict['iou_2d'])}
    scoreList = {}
    for seq_name in ['Kalman Filter', 'FtF', 'FtS', 'SEG-LSTM']:
        successRateList = []
        for threshold in thresholdSetOverlap:
            seqSuccessList = []
            for seq_iou_2d in success_Rate_List[seq_name]:
                seqSuccess = [score for score in seq_iou_2d if score > threshold]
                seqSuccessList.append(len(seqSuccess)/float(length))
            successRateList.append(seqSuccessList)
        scoreList[seq_name] = {}
        scoreList[seq_name]['successRateList'] = [np.mean(x) for x in successRateList]
        scoreList[seq_name]['AUC'] = np.sum(scoreList[seq_name]['successRateList'])/len(thresholdSetOverlap)
    title = 'success_2d'
    plot_graph_success(scoreList, title)

    ################################################################
    for seq_name in ['Kalman Filter', 'FtF', 'FtS', 'SEG-LSTM']:
        precisionRateList = []
        for threshold in thresholdSetError:
            seqPrecisionRateList = []
            for seq_errCenter in ErrCenter_List[seq_name]:
                seqPrecisions = [score for score in seq_errCenter if score < threshold]
                seqPrecisionRateList.append(len(seqPrecisions)/float(length))
            precisionRateList.append(seqPrecisionRateList)
        precisionRateList_plot = [np.mean(x) for x in precisionRateList]
        scoreList[seq_name] = {}
        scoreList[seq_name]['precisionRateList'] = [np.mean(x) for x in precisionRateList_plot]
        scoreList[seq_name]['p20'] = scoreList[seq_name]['precisionRateList'][20]
    title = 'precision_2d'
    plot_graph_precision(scoreList, title)
    plt.show()


def plot_graph_success(scoreList, title):
    plt.figure(num=0, figsize=(9, 6), dpi=70)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    rankList = sorted(scoreList, key=lambda x: (scoreList[x]['AUC']), reverse=True)
    for i in range(len(rankList)):
        ls = '-'
        if i % 2 == 1:
            ls = '--'
        successRateList_plot = scoreList[rankList[i]]['successRateList']
        ave = sum(successRateList_plot) / (len(successRateList_plot)) * 100
        if rankList[i] == 'SEG-LSTM':
            plt.plot(thresholdSetOverlap, successRateList_plot, c='r', label='{0} [{1:.3f}]'.format(rankList[i], ave), lw=8.0, ls='-')
        else:
            plt.plot(thresholdSetOverlap, successRateList_plot, c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format(rankList[i], ave), lw=2.0, ls=ls)

        #plt.title('Success plots')
        plt.rcParams.update({'axes.titlesize': 'large'})
        # plt.xlabel('IOU thresholds')
        plt.xticks(np.arange(thresholdSetOverlap[0], thresholdSetOverlap[len(thresholdSetOverlap)-1]+0.1, 0.1))
        plt.grid(color='#101010', alpha=0.5, ls=':')
        plt.legend(fontsize='medium')
        plt.savefig(os.path.join('../Experiments/figures', title+'.png'), dpi=140, bbox_inches='tight')
    return plt


def plot_graph_precision(scoreList, title, threshold_cm=20):
    plt.figure(num=1, figsize=(9, 6), dpi=70)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    rankList = sorted(scoreList, key=lambda x: (scoreList[x]['p20']), reverse=True)
    for i in range(len(rankList)):
        ls = '-'
        if i % 2 == 1:
            ls = '--'
        precisionRateList_plot = scoreList[rankList[i]]['precisionRateList']
        ave = precisionRateList_plot[threshold_cm]
        if rankList[i] == 'SEG-LSTM':
            plt.plot(thresholdSetError, precisionRateList_plot, c='r', label='{0} [{1:.3f}]'.format(rankList[i], ave), lw=8.0, ls='-')
        else:
            plt.plot(thresholdSetError, precisionRateList_plot, c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format(rankList[i], ave), lw=2.0, ls=ls)

        #plt.title('Precision plots')
        plt.rcParams.update({'axes.titlesize': 'large'})
        plt.xticks(np.arange(thresholdSetError[0], thresholdSetError[len(thresholdSetError) - 1], 10))
        plt.grid(color='#101010', alpha=0.5, ls=':')
        plt.legend(fontsize='medium')
        plt.savefig(os.path.join('../Experiments/figures', title+'.png'), dpi=140, bbox_inches='tight')

    return plt


if __name__ == '__main__':
    main()
