import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
import numpy as np
import re
import os

def print_confusion_as_csv(eng_abbrs, confusion_matrix):
    _confusion_matrix = confusion_matrix.tolist()

    LEN = len(eng_abbrs)
    print('-, ', end='')
    for abbr in eng_abbrs:
        print(abbr ,end=', ')
    print('')
    for i in range(LEN):
        print(eng_abbrs[i] ,end=', ')
        for j in range(LEN):
            print(_confusion_matrix[i][j] ,end=', ')
        print('')


def draw_auc(targets, predicts, savefig_filename='auc.png'):
    # print(savefig_filename)
    # print(targets)
    # print(predicts)
    fpr, tpr, threshold = roc_curve(targets, predicts)
    #print(fpr, tpr, threshold)

    auc1 = auc(fpr, tpr)
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(savefig_filename)
    plt.close()

def total_hit_topk(outputs, label, k):
    hit = 0
    for row in range(len(outputs)):
        # top k
        _, indices = outputs[row].topk(k, dim = 0, largest = True, sorted = True)
        for j in range(len(indices)):
            if label[row][indices[j].item()].item() == 1.0:
                hit += 1
                break
    return hit




def cal_confusion_matrix(targets, predicts):
    target_class , predict_class = [], []
    for t,p in zip(targets, predicts):
        target_class.append(t.index(max(t)))
        predict_class.append(p.index(max(p)))

    cm = confusion_matrix(target_class, predict_class)

    return np.matrix(cm)


def output_predict_result_sgfs(stat, eng_abbrs, tags_ch):
    TagCPattern = '[^AP]C\[([^\]]*)\]'
    for i , (p,t,sgf) in enumerate(zip(stat['predicts'],stat['target'], stat['sgf'])):
        comment = re.findall(TagCPattern, sgf)[-1]
        sgfstr = sgf.replace(comment, 'predict:{}  target:{}'.format(tags_ch[ p.index(max(p)) ] ,tags_ch[t.index(max(t))]))
        
        with open(os.path.join('../tempsgf',str(i)+'.sgf'), 'w') as fout:
            fout.write(sgfstr)

    
    