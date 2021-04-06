from train import Network


if __name__ == '__main__':
    train_network = Network()
    values = train_network.train()
    print(values)