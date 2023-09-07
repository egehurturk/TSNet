from imports import *
from utils import *
from models import *
from configuration import *

def main():
    train_configuration = asdict(TrainConfiguration())
    prepare(train_configuration)

    # Initialize the model with the training configurations
    model = DistNet(train_configuration)

    # Train the WaveNet with a single GPU (T4 instance) with the epochs specified in the training configuration
    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            log_every_n_steps=100,
            max_epochs=train_configuration['max_epochs'],
    )
    trainer.fit(model)
    # Save the checkpoints
    trainer.save_checkpoint(train_configuration['model'])

if __name__ == "__main__":
    main()
