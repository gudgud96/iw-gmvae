import torch
import torchvision
from torch import optim
from model import *
import json

with open('config.json') as f:
  args = json.load(f)

n_epochs = args["n_epochs"]
batch_size_train = args["batch_size_train"]
batch_size_test = args["batch_size_test"]
learning_rate = args["learning_rate"]
momentum = args["momentum"]
log_interval = args["log_interval"]

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_test, shuffle=True)

if args["name"] == "vae":
  model = VAE()
elif args["name"] == "iw-vae":
  model = IW_VAE(num_samples=args["num_samples"])
elif args["name"] == "gmvae":
  model = GMVAE()
else:
  model = IW_GMVAE(num_samples=args["num_samples"])
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

step = 0
for ep in range(1, n_epochs + 1):
    print("Epoch {} / {}".format(ep, n_epochs))
    
    for i, x in enumerate(train_loader):
        print("Batch {} / {}".format(i + 1, len(train_loader)), end="\r")
        optimizer.zero_grad()

        img, labels = x
        img, labels = img.cuda(), labels.cuda()
        img_hat, z, mu, var, log_logits_z, cls_z_prob = model(img)
        loss, recon_loss, kl_loss, clf_acc = model.loss_function(step, img_hat, img, mu, var, cls_z_prob, labels)
        step += 1

        loss.backward()
        optimizer.step()
    
    print("Loss: {:.4} Recon: {:.4} KL: {:.4} Acc: {:.4}".format(loss, recon_loss, kl_loss, clf_acc))
    
    batch_loss, batch_recon, batch_kl, batch_acc = 0, 0, 0, 0
    for i, x in enumerate(test_loader):
        print("Test Batch {} / {}".format(i + 1, len(test_loader)), end="\r")
        img, labels = x
        img, labels = img.cuda(), labels.cuda()
        img_hat, z, mu, var, log_logits_z, cls_z_prob = model(img)
        loss, recon_loss, kl_loss, clf_acc = model.loss_function(step, img_hat, img, mu, var, cls_z_prob, labels)
        batch_loss += loss.item() / len(test_loader)
        batch_recon += recon_loss.item() / len(test_loader)
        batch_kl += kl_loss.item() / len(test_loader)
        batch_acc += clf_acc / len(test_loader)

    print("Loss: {:.4} Recon: {:.4} KL: {:.4} Acc: {:.4}".format(batch_loss, batch_recon, batch_kl, batch_acc))

    torch.save(model.state_dict(), "{}.pt".format(args["name"]))