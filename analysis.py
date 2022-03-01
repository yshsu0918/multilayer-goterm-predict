import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc

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