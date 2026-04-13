import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys
from collections import OrderedDict
import scipy.stats
plt.switch_backend('agg')

NUM_BINS = 500
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1

LOG = './test_results/'

# === 修改点 1: 自动识别图例 (动态读取 LOG 文件夹内的文件前缀) ===
# 扫描形如 'rl_sac_trace_0.txt', 'llm_sac_trace_0.txt' 的文件
if os.path.exists(LOG):
    _detected_schemes = set([f.split('_')[0] for f in os.listdir(LOG) if f.endswith('.txt') and '_' in f])
    SCHEMES = sorted(list(_detected_schemes))
else:
    SCHEMES = ['rl']

# 根据识别到的前缀赋予图例标签，如果没有特殊定义，则直接转为大写 (如 'LLM')
_label_map = {'rl': 'Pure RL Baseline', 'llm': 'LLM-Augmented (Ours)'}
GLOBAL_LABELS = [_label_map.get(s, s.upper()) for s in SCHEMES]
# =========================================================

LW = 1.5

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0: return 0, 0, 0 # 防止空数组报错
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def inlist(filename, traces):
    ret = False
    for trace in traces:
        if trace in filename:
            ret = True
            break
    return ret

def bitrate_smo(outputs):
    # === 修改点 2: 移除函数内部写死的 SCHEMES 和 labels ===
    # SCHEMES = ['rl']
    # labels = ['Pure DQN Baseline']
    labels = GLOBAL_LABELS
    # ===================================================
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F']

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            # === 修改点 3: 严谨匹配文件前缀，防止 'rl' 匹配到其他单词 ===
            # if scheme in files:
            if files.startswith(scheme + '_'):
            # =======================================================
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_smo)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx % len(modern_academic_colors)],
            marker = markers[idx % len(markers)], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Bitrate Smoothness (mbps)')
    ax.set_ylabel('Video Bitrate (mbps)')
    
    # === 修改点 4: 解除写死的下边界限制，防止低码率数据超出画幅下方 ===
    # ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)
    if max_bitrate > 0:
        ax.set_ylim(0, max_bitrate * 1.1)
    # ==============================================================

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # === 修改点 5: loc='best' 让 matplotlib 自动寻找不遮挡数据的位置 ===
    # ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.legend(fontsize=12, ncol=1, edgecolor='white', loc='best')
    # ===============================================================
    
    ax.invert_xaxis()

    fig.savefig(outputs + '.png')
    plt.close()

def smo_rebuf(outputs):
    # SCHEMES = ['rl']
    # labels = ['Pure DQN Baseline']
    labels = GLOBAL_LABELS
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F']

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if files.startswith(scheme + '_'): # <--- 修改点 3 同上
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_smo)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx % len(modern_academic_colors)],
            marker = markers[idx % len(markers)], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Bitrate Smoothness (mbps)')
    
    # ax.set_ylim(0.05, max_bitrate + 0.05) # <--- 修改点 4 同上
    if max_bitrate > 0:
        ax.set_ylim(0, max_bitrate * 1.1)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left') # <--- 修改点 5 同上
    ax.legend(fontsize=12, ncol=1, edgecolor='white', loc='best')
    
    ax.invert_xaxis()
    ax.invert_yaxis()

    fig.savefig(outputs + '.png')
    plt.close()

def bitrate_rebuf(outputs):
    # SCHEMES = ['rl']
    # labels = ['Pure DQN Baseline']
    labels = GLOBAL_LABELS
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F']

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if files.startswith(scheme + '_'): # <--- 修改点 3 同上
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx % len(modern_academic_colors)],
            marker = markers[idx % len(markers)], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Video Bitrate (mbps)')
    
    # ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01) # <--- 修改点 4 同上
    if max_bitrate > 0:
        ax.set_ylim(0, max_bitrate * 1.1)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left') # <--- 修改点 5 同上
    ax.legend(fontsize=12, ncol=1, edgecolor='white', loc='best')
    
    ax.invert_xaxis()

    fig.savefig(outputs + '.png')
    plt.close()

def qoe_cdf(outputs):
    # SCHEMES = ['rl']
    # labels = ['Pure DQN Baseline']
    labels = GLOBAL_LABELS
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F']

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    plt.subplots_adjust(left=0.06, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        for files in os.listdir(LOG):
            if files.startswith(scheme + '_'): # <--- 修改点 3 同上
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        arr.append(float(sp[-1]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
        reward_all[scheme] = mean_arr

        if len(mean_arr) > 0: # 防空数据报错
            values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
            cumulative = np.cumsum(values)
            cumulative = cumulative / np.max(cumulative)
            ax.plot(base[:-1], cumulative, '-', \
                    color=modern_academic_colors[idx % len(modern_academic_colors)], lw=LW, \
                    label='%s: %.2f' % (labels[idx], np.mean(mean_arr)))

            print('%s, %.2f' % (scheme, np.mean(mean_arr)))
            
    ax.set_xlabel('QoE')
    ax.set_ylabel('CDF')
    ax.set_ylim(0., 1.01)
    
    # === 修改点 6: 解除 X 轴范围写死导致的 CDF 曲线被强制截断问题 ===
    # 原本为 ax.set_xlim(0., 1.8)，如果分数是 2.2 或者 -5 会全部被切掉出框
    # 去除这一行后，matplotlib 会根据数据的实际极值自动缩放 X 轴
    # ==============================================================

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower right')
    ax.legend(fontsize=12, ncol=1, edgecolor='white', loc='best') # 自动防遮挡

    fig.savefig(outputs + '.png')
    plt.close()

if __name__ == '__main__':
    bitrate_rebuf('baselines-br')
    smo_rebuf('baselines-sr')
    bitrate_smo('baselines-bs')
    qoe_cdf('baselines-qoe')