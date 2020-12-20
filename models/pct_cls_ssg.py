import torch.nn as nn
import torch.nn.functional as F
from pct_utils import TDLayer, PTBlock


class get_model(nn.Module):
    def __init__(self,num_class,N,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.input_mlp = nn.Sequential(
            nn.Conv1d(in_channel, 16, 1), 
            nn.BatchNorm1d(16), 
            nn.ReLU(), 
            nn.Conv1d(16, 32, 1), 
            nn.BatchNorm1d(32))
        self.TDLayer0 = TDLayer(npoint=N, input_dim=in_channel, out_dim=32, k=16)
        self.PTBlock0 = PTBlock(in_dim=32)

        self.TDLayer1 = TDLayer(npoint=int(N/4),input_dim=32, out_dim=64, k=16)
        self.PTBlock1 = PTBlock(in_dim=64)

        self.TDLayer2 = TDLayer(npoint=int(N/16),input_dim=64, out_dim=128, k=16)
        self.PTBlock2 = PTBlock(in_dim=128)

        self.TDLayer3 = TDLayer(npoint=int(N/64),input_dim=128, out_dim=256, k=16)
        self.PTBlock3 = PTBlock(in_dim=256)

        self.TDLayer4 = TDLayer(npoint=int(N/256),input_dim=256, out_dim=512, k=16)
        self.PTBlock4 = PTBlock(in_dim=512)


        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, inputs):
        B,_,_ = list(inputs.size())

        
        if self.normal_channel:
            l0_xyz = inputs[:, :3, :]
        else:
            l0_xyz = inputs
        l0_xyz, l0_points, l0_xyz_local, l0_points_local = self.TDLayer0(l0_xyz, inputs)
        l0_points = self.PTBlock0(l0_xyz, l0_points, l0_xyz_local, l0_points_local)
        
        l1_xyz, l1_points, l1_xyz_local, l1_points_local = self.TDLayer1(l0_xyz, l0_points)
        l1_points = self.PTBlock1(l1_xyz, l1_points, l1_xyz_local, l1_points_local)

        l2_xyz, l2_points, l2_xyz_local, l2_points_local = self.TDLayer2(l1_xyz, l1_points)
        l2_points = self.PTBlock2(l2_xyz, l2_points, l2_xyz_local, l2_points_local)

        l3_xyz, l3_points, l3_xyz_local, l3_points_local = self.TDLayer3(l2_xyz, l2_points)
        l3_points = self.PTBlock3(l3_xyz, l3_points, l3_xyz_local, l3_points_local)

        l4_xyz, l4_points, l4_xyz_local, l4_points_local = self.TDLayer4(l3_xyz, l3_points)
        l4_points = self.PTBlock4(l4_xyz, l4_points, l4_xyz_local, l4_points_local)

        l4_points = l4_points.mean(dim=-1)

        x = l4_points.view(B, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        #x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc2(x)
        x = F.log_softmax(x, -1)


        return x, l4_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
