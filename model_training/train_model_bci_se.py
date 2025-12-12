from omegaconf import OmegaConf
from FastBCI import BrainToTextDecoder_Trainer

args = OmegaConf.load('fastbci_se_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
