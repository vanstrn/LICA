REGISTRY = {}

from .rnn_agent import RNNAgent,CRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["crnn"] = CRNNAgent
