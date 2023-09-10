import torch

class Accuracy:
    
    def calculate_iou(self,y_true, y_pred):
        intersection = torch.logical_and(y_true, y_pred).sum(dim=(1, 2)).float()
        union = torch.logical_or(y_true, y_pred).sum(dim=(1, 2)).float()
        iou = intersection / union
        return iou.mean()

    def calculate_pixel_accuracy(self,y_true, y_pred):
        correct_pixels = torch.logical_and(y_true, y_pred).sum(dim=(1, 2)).float()
        total_pixels = y_true.sum(dim=(1, 2)).float()
        pixel_accuracy = (correct_pixels / total_pixels).mean()
        return pixel_accuracy

    def calculate_dice_coefficient(self,y_true, y_pred):
        intersection = (y_true * y_pred).sum(dim=(1, 2)).float()
        union = y_true.sum(dim=(1, 2)).float() + y_pred.sum(dim=(1, 2)).float()
        dice_coefficient = (2.0 * intersection) / union
        return dice_coefficient.mean()

    def calculate_f1_score(self, y_true, y_pred):
        true_positives = torch.logical_and(y_true, y_pred).sum(dim=(1, 2)).float()
        false_positives = torch.logical_and(~y_true.bool(), y_pred).sum(dim=(1, 2)).float()
        false_negatives = torch.logical_and(y_true, ~y_pred.bool()).sum(dim=(1, 2)).float()

        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1_score.mean()