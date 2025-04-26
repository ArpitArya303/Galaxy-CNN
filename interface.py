# interface.py

from model import GalaxyCNN as TheModel
from train import train_model as the_trainer
from predict import predict_galaxy as the_predictor
from dataset import get_dataset as TheDataset
from dataset import get_loader as the_dataloader