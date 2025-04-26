# interface.py

from model import GalaxyCNN as TheModel
from train import train_model as the_trainer
from predict import classify_galaxies as the_predictor
from dataset import get_dataset as TheDataset
from dataset import get_loader as the_dataloader
from config import batch_size as the_batch_size
from config import epochs as total_epochs
