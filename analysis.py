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