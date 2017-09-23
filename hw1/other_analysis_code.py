import matplotlib.pyplot as plt

def plot_example_vs_accuracy(x,y):

    plt.figure()
    plt.title("Training Accuracy  VS  K")
    plt.xlabel("K (k-nearest neighbor)")
    plt.ylabel("Training Accuracy")

    plt.plot(x, y, 'bo', x, y, 'k')

    plt.legend(loc="best")
    return plt

if __name__ == "__main__":

    x = [500, 1000, 5000, 10000, 20000, 30000, 40000, 50000]
    y = [0.8311, 0.8758, 0.94, 0.9544, 0.963, 0.97, 0.9714, 0.9727]

    x = [1,3,5,7,9,11]
    y = [1.000000,0.986340,0.979440,0.976120,0.974000,0.971560]

    plot_example_vs_accuracy(x,y).show()
