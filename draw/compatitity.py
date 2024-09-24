import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pylustrator
import os
# pylustrator.start()
# 数据


def plot_compatity():
    data = {
        'Models': ['N/A', 'SmallBERT', 'MediumBERT', 'TinyBERT', 'DistilBERT', 'BERT', 'RoBERTa'],
        'AUC_MovieLens': [0.8410, 0.8375, 0.8410, 0.8417, 0.8447, 0.8438, 0.8392],
        'LogLoss_MovieLens': [0.4036, 0.4071, 0.4031, 0.4074, 0.3983, 0.4019, 0.4038],
        'AUC_Bookcrossing': [0.7888, 0.7902, 0.7905, 0.7905, 0.7913, 0.7909, 0.7904],
        'LogLoss_Bookcrossing': [0.4932, 0.4929, 0.4915, 0.4910, 0.4906, 0.4930, 0.4917]
    }

    # 将数据转换为 pandas DataFrame
    x = data['Models']
    y1 = data['AUC_MovieLens']
    y2 = data['LogLoss_MovieLens']
    y3 = data['AUC_Bookcrossing']
    y4 = data['LogLoss_Bookcrossing']
    plt.rcParams['figure.figsize'] = (16.0, 8)#设置图片大小

    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 15  # 设置基础字体大小
    plt.rcParams['axes.titlesize'] = 19  # 设置标题字体大小
    plt.rcParams['axes.labelsize'] = 17  # 设置坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 19  # 设置 x 轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 19  # 设置 y 轴刻度标签字体大小
    plt.rcParams['legend.fontsize'] = 15  # 设置图例字体大小
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.set_title('MovieLens-1M')
    ax1.set_xlabel("PLMs")
    ax1.set_ylabel("AUC")
    width = range(len(x))
    line1 = ax1.plot(width, y1, c="c", label="AUC", marker="o")
    ax1.set_xticks(range(len(x)), x)
    ax1_1 = ax1.twinx()
    ax1_1.set_ylabel('LogLoss')
    line2 = ax1_1.plot(width, y2, c="orange", marker="D", linestyle="--", label="LogLoss")
    lines1 = line1 + line2
    labs1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labs1, loc="best")

    ax2.set_title('BookCrossing')
    ax2.set_xlabel("PLMs")
    ax2.set_ylabel("AUC")
    line3 = ax2.plot(width, y3, c="c", label="AUC", marker="o")
    ax2.set_xticks(range(len(x)), x)
    ax2_1 = ax2.twinx()
    ax2_1.set_ylabel('LogLoss')
    line4 = ax2_1.plot(width, y4, c="orange", marker="D", linestyle="--", label="LogLoss")
    lines2 = line3 + line4
    labs2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labs2, loc="best")

    plt.tight_layout()
    #保存桌面
    plt.savefig(r'C:\Users\Administrator\Desktop/fig/compatibility.pdf', format='pdf')
    plt.show()
    
 

    # plt.savefig('./pics/compatibility.pdf', format='pdf')
plot_compatity()