import torch

checkpoint = torch.load('C:/Users/seok436/PycharmProjects/packnet-sfm/configs/fm_depth.pth')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.cuda()
model.eval()
