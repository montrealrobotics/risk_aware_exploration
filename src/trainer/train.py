import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

import tqdm
import wandb
from src.models.risk_models import * 
from src.datasets.risk_datasets import *


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./traj",
        help="the name of this experiment")
    parser.add_argument("--env", type=str, default="SafetyCarButton1Gymnasium-v0",
        help="the name of this experiment")
    parser.add_argument("--dataset_type", type=str, default="state_risk",
        help="what dataset to use state or state_action?")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
        help="learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=1000, 
        help="batch size for the stochastic gradient descent" ) 
    parser.add_argument("--validate-every", type=int, default=2000,
        help="validate every x SGD steps")
    parser.add_argument("--num_iterations", type=int, default=100, 
        help="number of times to go over the entire dataset during training")
    parser.add_argument("--obs-size", type=int, default=72, 
        help="size of the first layer of the mlp")
    parser.add_argument("--fc1_size", type=int, default=128, 
        help="size of the first layer of the mlp")
    parser.add_argument("--fc2_size", type=int, default=128,
        help="size of the second layer of the mlp")
    parser.add_argument("--fc3_size", type=int, default=128,
        help="size of the third layer of the mlp")
    parser.add_argument("--fc4_size", type=int, default=128,
        help="size of the fourth layer of the mlp")
    parser.add_argument("--lr_schedule", type=float, default=0.99,
        help="schedule for the learning rate decay " ) 
    parser.add_argument("--weight", type=float, default=1.0, 
        help="weight for the 1 class in BCE loss")
    parser.add_argument("--model-type", type=str, default="mlp",
                    help="which network to use for the risk model")
    parser.add_argument("--fear-radius", type=int, default=5, help="radius around the dangerous objects to consider fearful. ")
    parser.add_argument("--fear-clip", type=float, default=1000., help="radius around the dangerous objects to consider fearful. ")
    parser.add_argument("--risk-type", type=str, default="discrete",
                    help="what kind of risk model to train")
    parser.add_argument("--quantile-size", type=int, default=4, help="size of the risk quantile ")
    parser.add_argument("--quantile-num", type=int, default=5, help="number of quantiles to make")

    return parser.parse_args()


args = parse_args()

run = wandb.init(config=vars(args), entity="manila95",
                project="risk-aware-exploration",
                monitor_gym=True,
                sync_tensorboard=True, save_code=True)


f_obs = torch.load("./data/pointgoal/obs.pt")
f_risks = torch.load("./data/pointgoal/risks.pt")

device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

dataset = RiskyDataset(f_obs, None, f_risks, False, "quantile",
                            fear_clip=None, fear_radius=40, one_hot=True, quantile_size=4, quantile_num=10)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=10, generator=torch.Generator(device='cpu'))

criterion = nn.NLLLoss()
model = BayesRiskEst(108, out_size=10)
model.to(device)
opt = optim.Adam(model.parameters(), lr=args.learning_rate)

for i in tqdm.tqdm(range(200)):
    net_loss = 0
    for batch in dataloader:
        pred = model(batch[0].to(device))
        loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        net_loss += loss.item()
    print("Epoch %d | Loss = %.3f"%(i, net_loss))
    wandb.log({"Loss": net_loss})

torch.save(model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
wandb.save("risk_model.pt")
model.eval()