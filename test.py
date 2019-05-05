
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





def example_0():

    # learns to repeat simple sequence from random inputs

    np.random.seed(0)



    # parameters for input data dimension and lstm cell count

    mem_cell_ct = 100

    x_dim = 50

    gru_param = GRUParam(mem_cell_ct, x_dim)

    gru_net = GRUNetwork(gru_param)
    y_list = [-0.5, 1, 0.3, 0.6, 0.2, -0.1, 0.3, 0.8]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(1000):

        print("iter", "%2s" % str(cur_iter), end=": ")

        for ind in range(len(y_list)):

            gru_net.x_list_add(input_val_arr[ind])



        print("y_pred = [" +

              ", ".join(["% 2.5f" % gru_net.gru_node_list[ind].state.h[0] for ind in range(len(y_list))]) +

              "]", end=", ")



        loss = gru_net.y_list_is(y_list, ToyLossLayer)

        print("loss:", "%.3e" % loss)

        gru_param.apply_diff(lr=0.01)

        gru_net.x_list_clear()





if __name__ == "__main__":

    example_0()

