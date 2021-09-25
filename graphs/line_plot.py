import matplotlib.pyplot as plt

def plot(plots, labels, xlabel, ylabel, title):
    for index, metric in enumerate(plots):
        plt.plot(metric[0], metric[1], label = labels[index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1,1), loc = 'upper left')
    plt.title(title)
    plt.savefig(f'./images/{title} Graph.jpg', bbox_inches='tight')
    plt.show()
    
    
