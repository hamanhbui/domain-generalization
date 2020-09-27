from algorithms.Metric.src.models import alexnet
from algorithms.Metric.src.models import resnet
from algorithms.Metric.src.models import caffenet

nets_map = {
    'alexnet': alexnet.alexnet,
    'resnet18': resnet.resnet18,
    'resnet50': resnet.resnet50,
    'caffenet': caffenet.caffenet
}

def get_model(name):
    if name not in nets_map:
        raise ValueError('Name of model unknown %s' % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn