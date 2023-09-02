import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.p.feature_size = 4
        self.p.grid_x = 320
        self.p.grid_y = 320
        
    def forward(self, feature, ground_truth_instance):
        real_batch_size = feature.size(0)
        
        feature_map = feature.view(real_batch_size, self.p.feature_size, 1, self.p.grid_y * self.p.grid_x)
        feature_map = feature_map.expand(real_batch_size, self.p.feature_size, self.p.grid_y * self.p.grid_x, self.p.grid_y * self.p.grid_x).detach()
        
        point_feature = feature.view(real_batch_size, self.p.feature_size, self.p.grid_y * self.p.grid_x, 1)
        point_feature = point_feature.expand(real_batch_size, self.p.feature_size, self.p.grid_y * self.p.grid_x, self.p.grid_y * self.p.grid_x)
        
        distance_map = (feature_map - point_feature) ** 2
        distance_map = torch.norm(distance_map, dim=1).view(real_batch_size, 1, self.p.grid_y * self.p.grid_x, self.p.grid_y * self.p.grid_x)

        # Calculate same instance loss (sisc_loss)
        sisc_loss = torch.sum(distance_map[ground_truth_instance == 1]) / torch.sum(ground_truth_instance == 1)

        # Calculate different instance loss (disc_loss)
        disc_loss = self.p.K1 - distance_map[ground_truth_instance == 2]
        disc_loss[disc_loss < 0] = 0
        disc_loss = torch.sum(disc_loss) / torch.sum(ground_truth_instance == 2)

        total_loss = sisc_loss + disc_loss
        return total_loss
