# %%
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    one = np.load('./record/one_position.npy')
    one_move = np.load('./record/one_position+move_feature.npy')
    two = np.load('./record/two_position.npy')
    two_move = np.load('./record/two_position+move_feature.npy')
    sub = np.load('./record/subsampling.npy')
    select = np.load('./record/select_move1.npy')
    x = np.arange(1, len(one)+1)
    plt.figure(figsize=(10, 10))
    plt.title('Term Model(top3)')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.yticks(np.arange(0, 1, step=0.05))
    plt.plot(x, one, 'red', label='one_position')
    plt.plot(x, one_move, 'orange', label='one_position+move_feature')
    plt.plot(x, two, 'yellow', label='two_position')
    plt.plot(x, two_move, 'green', label='two_position+move_feature')
    plt.plot(x, sub, 'blue', label='subsampling')
    plt.plot(x, select, 'aqua', label='select_move')
    plt.legend(loc='best')
    plt.show()



# %%