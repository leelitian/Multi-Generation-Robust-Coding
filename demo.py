import torch
from PIL import Image
from torchvision import transforms
from models.joinmbt import JoinMbt

torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = 'cpu'
    checkpoint = torch.load('checkpoint_600.pth.tar', map_location=device)
    print(checkpoint["state_dict"].keys())

    net = JoinMbt(192, 192).to(device).eval()
    net.load_state_dict(checkpoint["state_dict"])

    img = Image.open('kodim18.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # codec
        out = net.compress(x, 'foo')
        rec = net.decompress('foo')
        rec = transforms.ToPILImage()(rec['x_hat'].clamp(0, 1).squeeze().cpu())
        rec.save('./images/codec.png', format="PNG")

        # inference
        out = net(x)
        rec = out['x_hat'].clamp(0, 1)
        rec = transforms.ToPILImage()(rec.squeeze().cpu())
        rec.save('./images/infer.png', format="PNG")

        print('saved in ./images')
