import matplotlib.pyplot as plt

def load(bs, hl, dr):
    name = str(bs) + '-' + str(hl) + '-0.' + str(dr)
    with open('./best_models/'+name+'.txt', 'r') as f:
        val_loss = f.readline()[11:-2]
        val_loss = val_loss.replace(',', '')
        val_loss = val_loss.split(' ')
        val_loss = list(map(float, val_loss))

        val_acc = f.readline()[10:-2]
        val_acc = val_acc.replace(',', '')
        val_acc = val_acc.split(' ')
        val_acc = list(map(float, val_acc))

        loss = f.readline()[7:-2]
        loss = loss.replace(',', '')
        loss = loss.split(' ')
        loss = list(map(float, loss))

        acc = f.readline()[6:-2]
        acc = acc.replace(',', '')
        acc = acc.split(' ')
        acc = list(map(float, acc))

    return name, val_loss, val_acc, loss, acc

def load_path(path):
    with open('./' + path + '.txt', 'r') as f:
        val_loss = f.readline()[11:-2]
        val_loss = val_loss.replace(',', '')
        val_loss = val_loss.split(' ')
        val_loss = list(map(float, val_loss))

        val_acc = f.readline()[10:-2]
        val_acc = val_acc.replace(',', '')
        val_acc = val_acc.split(' ')
        val_acc = list(map(float, val_acc))

        loss = f.readline()[7:-2]
        loss = loss.replace(',', '')
        loss = loss.split(' ')
        loss = list(map(float, loss))

        acc = f.readline()[6:-2]
        acc = acc.replace(',', '')
        acc = acc.split(' ')
        acc = list(map(float, acc))

    return val_loss, val_acc, loss, acc

def plot_acc(nr_fig, name, acc, val_acc):
    plt.subplot(4, 3, nr_fig)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title(name, size=8)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')


def plot_loss(nr_fig, name, loss, val_loss):
    plt.subplot(4, 3, nr_fig)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(name, size=8)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')

def plot_best_models():
    bs = [32, 64, 128]
    hl = [64, 128, 256, 512]
    dr = [2, 4, 5]

    figure = 1
    for _bs in bs:
        fig = plt.figure(figure)
        fig.suptitle('Tr-Albastru Ts-Portocaliu - ' + str(_bs) + ' BS')
        plot_number = 1
        for _hl in hl:
            for _dr in dr:
                name, val_loss, val_acc, loss, acc = load(_bs, _hl, _dr)
                plot_acc(plot_number, name, acc, val_acc)
                plot_number += 1

        figure += 1

def plot_random(path):
    val_loss, val_acc, loss, acc = load_path(path)

    fig1 = plt.figure(1)
    fig1.suptitle('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    fig2 = plt.figure(2)
    fig2.suptitle('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

if __name__ == '__main__':
    #plot_best_models()
    plot_random('best_models/single_models/Conv128')

    plt.show()