from lightning_module import MuffinChihuahuaLightningModule
from interfaces import MuffinChihuahuaDataModule
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

def main():
    # Init logger
    # wandb.init(project="MUFFIN-CHIHUAHUA")
    # 
    
    # Init data module
    data_module = MuffinChihuahuaDataModule(data_dir="data",
                                            batch_size=32,
                                            numworkers=8)
    # data_module.prepare_data()
    # data_module.setup()
    
    # Init model
    model = MuffinChihuahuaLightningModule()

    # Init trainer
    trainer = Trainer(fast_dev_run=False, max_epochs=10)
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)


if __name__ == '__main__':
    main()