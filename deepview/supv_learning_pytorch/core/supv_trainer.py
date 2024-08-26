
#
# import sys
# sys.path.append("..")
# from deepview.supv_learning_pytorch.models.cnn import CNN  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.dcl import DeepConvLSTM  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.lstm import LSTM  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.dcl_v3 import DeepConvLSTM3  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.dcl_sa import DeepConvLSTMSelfAttn  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.resnet_l_sa import resnet_lstm_selfattn  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.transformer import Transformer  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.cnn_ae_v5 import CNN_AE5  # import classes from models.init.py
# from deepview.supv_learning_pytorch.models.cnn_ae_v6 import CNN_AE6  # import classes from models.init.py

from ..models.cnn import CNN  # import classes from models.init.py
from ..models.dcl import DeepConvLSTM  # import classes from models.init.py
from ..models.lstm import LSTM  # import classes from models.init.py
from ..models.dcl_v3 import DeepConvLSTM3  # import classes from models.init.py
from ..models.dcl_sa import DeepConvLSTMSelfAttn  # import classes from models.init.py
from ..models.resnet_l_sa import resnet_lstm_selfattn  # import classes from models.init.py
from ..models.transformer import Transformer  # import classes from models.init.py
from ..models.cnn_ae_v5 import CNN_AE5  # import classes from models.init.py
from ..models.cnn_ae_v6 import CNN_AE6  # import classes from models.init.py


def setup_model(cfg):
    # model configuration
    if cfg.model.model_name == "cnn": # CNN
        model = CNN(cfg)
    elif cfg.model.model_name == 'lstm': # LSTM
        model = LSTM(cfg)
    elif cfg.model.model_name == 'dcl': # DCL
        model = DeepConvLSTM(cfg)
    elif cfg.model.model_name == 'dcl-v3': # Mixup After LSTM layer
        model = DeepConvLSTM3(cfg)
    elif cfg.model.model_name == 'dcl-sa': # DCLSA
        model = DeepConvLSTMSelfAttn(cfg)
    elif cfg.model.model_name == 'resnet-l-sa': # DCLSA-RN (ResNet version of DCLSA)
        model = resnet_lstm_selfattn(cfg)
    elif cfg.model.model_name == 'transformer':
        model = Transformer(cfg)
    elif cfg.model.model_name in ['cnn-ae']: # CNN_AE5 for unsupervised pre-training
        model = CNN_AE5(cfg)
    elif cfg.model.model_name in ["cnn-ae-wo"]: # CNN_AE6 for hyperparameter tuning
        model = CNN_AE6(cfg)
    else:
        raise Exception(
            f"cfg.model.model_name: {cfg.model.model_name} is not appropriate.")

    return model