import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loadcsv(dirs):
    return [np.loadtxt(p, str, delimiter=',') for p in dirs]


def getMetricsData(rawdata, metrics_index, epoch):
    return [d[1:epoch + 1, metrics_index].astype(np.float) for d in rawdata]


def main(dirs, labels, metrics_index, ylabel, epoch=150,
         save_path='', xticks=[0, 50, 100, 150], yticks=[0.0, 0.2, 0.4, 0.6, 0.8],
         ylim=(0, 0.8), xlim=None, xlabel="epoch", cut=0):
    color = ["r", "b", 'g', 'y', 'k', 'c','darkorange' , 'gold', 'lime', 'm', 'pink', 'saddlebrown', 'deepskyblue',
             'mediumslateblue',  'orangered']
    rawdata = loadcsv(dirs)
    datas = getMetricsData(rawdata, metrics_index, epoch)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    x = np.arange(epoch)
    for i, y in enumerate(datas):
        plt.plot(x[cut:], y[cut:], label=labels[i], c=color[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # dirs = [
    #     r"G:\yolov8-result\org\results.csv",
    #     # r"G:\yolov8-result\org-bn\results.csv",
    #     # r"G:\yolov8-result\org-w\results.csv",
    #     r"G:\yolov8-result\plus\results.csv",
    #     r"G:\yolov8-result\plus+\results.csv",
    #     r"G:\yolov8-result\ultra-\results.csv",
    #     r"G:\yolov8-result\DYS\results.csv",
    #     r"G:\yolov8-result\cpca\results.csv",
    #     r"G:\yolov8-result\GD\results.csv",
    #     r"G:\yolov8-result\fast\results.csv",
    #     r"G:\yolov8-result\ghost\results.csv",
    #     r"G:\yolov8-result\yolov5\results.csv",
    #     r"G:\yolov8-result\SPPFCSPC\results.csv",
    #     r"G:\yolov8-result\EMA\results.csv",
    #     r"G:\yolov8-result\MSLAG\results.csv",
    #     r"G:\yolov8-result\MSLAP\results.csv",
    # ]
    # labels = [
    #     'org(yolov8s)',
    #     # 'org-bn(yolov8s)','org-w(yolov8s)',
    #     'plus(cpca+CARAFE+wiou)', 'plus+(LSK+DySample+wiou)',
    #     'ultra(DCNv3+DySample+wiou)', 'DYS(DySample)', 'cpca',
    #     'Gather & Distribute', 'fast', 'ghost', 'yolov5', 'SPPFCSPC', 'EMA', 'MSLAG', 'MSLAP'
    # ]
    dirs = [

        r"G:\yolov8-result\v8s-aod6-impd7-150\results.csv",
        # r"G:\yolov8-result\v8s-aod6-150\results.csv",
        #       r"G:\yolov8-result\v8s-dys\results.csv",
        # r"G:\yolov8-result\v8s-aod6-AD150\results.csv",
        # r"G:\yolov8-result\v8s-ema-32\results.csv",
        r"G:\yolov8-result\v8s-4\results.csv",
    ]
    labels = ['YOLOv8s-DAE', 'YOLOv8s'
         # 'improved', 'DySample', 'ADown','EMA','YOLOv8s'
    ]
    # 4 P, 5 R,6  mAP50, 7 mAP50-95
    p = [0.2, 0.4, 0.6, 0.8]
    # map50 = [0.4,0.6,0.8,1.0]
    map50 = [0.4,0.6,0.8,1.0]
    map5095 = [0.3, 0.35,0.4, 0.45,0.5]
    r = [0.4,0.45,0.5,0.55,0.6,0.65]
    main(dirs, labels, metrics_index=6, ylabel="mAP@.5", epoch=150,
         save_path='G:\yolov8-result\metrics\mAP50-new6.jpg',
         xlim=(0,150), xticks=[0,100,150],
         yticks=map50,ylim=(0.4, 1.0),
         cut=0)
'''
['                  epoch' '         train/box_loss'
 '         train/cls_loss' '         train/dfl_loss'
 '   metrics/precision(B)' '      metrics/recall(B)'
 '       metrics/mAP50(B)' '    metrics/mAP50-95(B)'
 '           val/box_loss' '           val/cls_loss'
 '           val/dfl_loss' '                 lr/pg0'
 '                 lr/pg1' '                 lr/pg2']

Process finished with exit code 0

'''
