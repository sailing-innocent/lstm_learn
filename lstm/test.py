import numpy as np

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

class OneShotLossLayer:
    """
    Computes the distance between two one-shot encoding vector
    """
    @classmethod
    def loss(self, pred, label):
        length = pred.shape[0]
        # print(length)
        sumsq = 0
        for i in range(length):
            sumsq += (pred[i]-label[i]) ** 2
        
        return sumsq
    @classmethod
    def bottom_diff(self, pred, label):
        sumsq = 0
        length = pred.shape[0]
        for i in range(length):
            sumsq += 2 * (pred[i]-label[i])
        return sumsq

def testOneShotLoss():
    one = np.array([0.0,1.0,0.0])
    two = np.array([0.0,0.0,1.0])
    loss_layer = OneShotLossLayer()
    print(loss_layer.loss(one, two))

def debugPred(y_pred):
    print(y_pred)

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    # mem_cell_ct = 100
    # x_dim = 50
    mem_cell_ct = 3
    x_dim = 3
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    # y_list = [-0.5, 0.2, 0.1, -0.5, 0.3, -0.2, 0.4]
    # input_val_arr = [np.random.random(x_dim) for _ in y_list]
    y_list = np.array([[0,1,0],[1,0,0],[0,1,0]])
    input_val_arr = np.array([[0,1,0],[1,0,0],[0,1,0]])

    every_print_loss = 100

    for cur_iter in range(2000):
        
        for ind in range(len(y_list)):
            # print(input_val_arr[ind].shape)
            lstm_net.x_list_add(input_val_arr[ind])
        

        # loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        loss = lstm_net.y_list_is(y_list, OneShotLossLayer)
        
        if (cur_iter % every_print_loss == 0):
            print("iter", "%2s" % str(cur_iter), end=": ")
            print("loss:", "%.3e" % loss)
    
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()
        
        # debugPred([lstm_net.lstm_node_list[ind].state.h for ind in range(len(y_list))])


    """
    print("TESTING ++++++++++++++++++++++++++++++++++++++++++++++")
    ylen = len(y_list) - 1
    test_val_arr = [input_val_arr[i] + 0.01 * np.ones(x_dim) for i in range(ylen)]
    for ind in range(ylen):
        lstm_net.x_list_add(test_val_arr[ind])


    print("y_pred = [" +
        ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(ylen)]) +
        "]", end=", ")
    """

if __name__ == "__main__":
    example_0()
    # testOneShotLoss()

