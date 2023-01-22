from lightning_module import MuffinChihuahuaLightningModule
from interfaces import MuffinChihuahuaDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # Init logger
    # wandb.init(project="MUFFIN-CHIHUAHUA")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    wandb_logger = WandbLogger(name='resnet18_22Jan_2', project="MUFFIN-CHIHUAHUA", log_model=True, )
    
    # Init data module
    data_module = MuffinChihuahuaDataModule(data_dir="data",
                                            batch_size=32,
                                            numworkers=36)
    # data_module.prepare_data()
    # data_module.setup()
    
    # Init model
    model = MuffinChihuahuaLightningModule()

    # Init trainer
    trainer = Trainer(fast_dev_run=False, max_epochs=10, accelerator='gpu', logger=wandb_logger, callbacks=[checkpoint_callback])
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)


if __name__ == '__main__':
    main()