import torch
from PySide6.QtCore import QCoreApplication, QObject, Signal

from deepview.clustering_pytorch.datasets.factory import generate_dataloader
from deepview.clustering_pytorch.nnet.common_config import get_model
from deepview.gui.supervised_cl.train.utils import (
    adjust_learning_rate,
    evaluate,
    get_scl_criterion_opt,
    get_window_data_scl,
    load_model_parameters,
    train,
)


class Scl2ClWorker(QObject):
    finished = Signal(tuple)  # Signal to indicate the task is finished
    progress = Signal(int)    # Signal to indicate progress
    stopped = Signal()

    def __init__(self, full_model_path, net_type, data, data_length, column_names, batch_size, aug1, aug2):
        super().__init__()
        self.full_model_path = full_model_path
        self.net_type = net_type
        self.data = data
        self.data_length = data_length
        self.column_names = column_names
        self.batch_size = batch_size
        self.aug1 = aug1
        self.aug2 = aug2
        self._is_running = True

    def run(self):
        selected_data, label_flag, timestamp, label = get_window_data_scl(
            self.data, self.column_names, self.data_length
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = generate_dataloader(
            selected_data, label, timestamp, self.batch_size,
            True, device, label_flag, self.aug1, self.aug2
        )

        num_channel = selected_data.shape[-1]

        model = get_model(
            p_backbone=self.net_type,
            p_setup='simclr',
            num_channel=num_channel,
            data_len=self.data_length
        ).to(device)

        loaded_model = load_model_parameters(model, self.full_model_path, device)
        loaded_model, criterion, optimizer = get_scl_criterion_opt(loaded_model, device)

        method = 'SimCLR'
        nepochs = 5

        for epoch in range(1, nepochs + 1):
            if not self._is_running:
                self.stopped.emit()
                return
            adjust_learning_rate(optimizer, epoch, nepochs)
            loss = train(train_loader, model, method, criterion, optimizer, epoch, nepochs, device)
            print(f'SimCLR loss of the {epoch}-th training epoch is : {loss}')
            self.progress.emit(int((epoch / nepochs) * 100))
            QCoreApplication.processEvents()

        repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR = evaluate(model, train_loader, device)

        method = 'Supervised_SimCLR'
        for epoch in range(1, nepochs + 1):
            if not self._is_running:
                self.stopped.emit()
                return
            loss = train(train_loader, model, method, criterion, optimizer, epoch, nepochs, device)
            print(f'Supervised_SimCLR loss of the {epoch}-th training epoch is : {loss}')
            self.progress.emit(int((epoch / nepochs) * 100))
            QCoreApplication.processEvents()

        repre_tsne_CLR, _, _ = evaluate(model, train_loader, device)

        self.finished.emit((repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR, repre_tsne_CLR))

    def stop(self):
        self._is_running = False