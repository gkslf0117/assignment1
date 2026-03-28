import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random

# -------------------------------
# 1. 모델 정의 (CNN)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = None  # adaptive
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# -------------------------------
# 2. 공격 함수들
# -------------------------------

# Targeted FGSM
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv - eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# Untargeted FGSM
def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# Targeted PGD
def pgd_targeted(model, x, target, k=40, eps=0.3, eps_step=0.01):
    k = int(k)
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv - eps_step * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

# Untargeted PGD
def pgd_untargeted(model, x, label, k=40, eps=0.3, eps_step=0.01):
    k = int(k)
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + eps_step * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

# -------------------------------
# 3. 데이터셋
# -------------------------------
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1, shuffle=False)
    return train_loader, test_loader

def get_cifar10_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform),
        batch_size=1, shuffle=False)
    return train_loader, test_loader


# -------------------------------
# 4. 학습 함수
# -------------------------------
def train(model, device, train_loader, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


# -------------------------------
# 5. 공격 테스트 
# -------------------------------
def test_attack(model, device, test_loader, attack_fn, attack_name,
                eps=0.1, num_samples=5, dataset_name='MNIST'):

    os.makedirs('results', exist_ok=True)
    model.eval()

    success_count = 0
    total_samples = 0
    max_samples = 100

    for i, (data, target) in enumerate(test_loader):

        if total_samples >= max_samples:
            break

        data, target = data.to(device), target.to(device)

        # 원래 맞은 샘플만 공격
        output = model(data)
        init_pred = output.argmax(dim=1)

        if init_pred.item() != target.item():
            continue

        total_samples += 1

        # -------------------------
        # 공격 수행
        # -------------------------
        
        # 'Untargeted'가 이름에 포함되어 있는지 먼저 확인
        if 'Untargeted' in attack_name:
            if 'PGD' in attack_name:
                adv_data = attack_fn(
                    model, data, target, k=40, eps=eps, eps_step=eps/4
                )
            else:
                adv_data = attack_fn(
                    model, data, target, eps
                )
            pred = model(adv_data).argmax(dim=1)
            
            # 원래 정답과 달라지면 공격 성공
            if pred.item() != target.item():
                success_count += 1

        else: # Targeted 공격인 경우
            target_class = torch.randint_like(target, 0, 10)
            while target_class.item() == target.item():
                target_class = torch.randint_like(target, 0, 10)

            if 'PGD' in attack_name:
                adv_data = attack_fn(
                    model, data, target_class, k=40, eps=eps, eps_step=0.01
                )
            else:
                adv_data = attack_fn(
                    model, data, target_class, eps
                )
            pred = model(adv_data).argmax(dim=1)
            
            # 목표했던 클래스(target_class)로 예측하면 공격 성공
            if pred.item() == target_class.item():
                success_count += 1



        # -------------------------
        # 시각화
        # -------------------------
        if i < num_samples:

            fig, axs = plt.subplots(1, 3, figsize=(10, 3))

            # MNIST (grayscale)
            if data.shape[1] == 1:

                axs[0].imshow(
                    data[0].cpu().squeeze(),
                    cmap='gray'
                )

                axs[1].imshow(
                    adv_data[0].cpu().squeeze(),
                    cmap='gray'
                )

                perturb = (
                    adv_data - data
                ).cpu().squeeze()

                axs[2].imshow(
                    perturb*10,
                    cmap='gray'
                )

            # CIFAR10 (RGB)
            else:

                axs[0].imshow(
                    data[0].cpu().permute(1, 2, 0)
                )

                axs[1].imshow(
                    adv_data[0].cpu().permute(1, 2, 0)
                )

                perturb = (
                    adv_data - data
                )[0].cpu().permute(1, 2, 0)

                axs[2].imshow(
                    perturb*10
                )

            axs[0].set_title(f"Orig: {target.item()}")
            axs[1].set_title(f"Adv: {pred.item()}")
            axs[2].set_title("Perturbation")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()

            plt.savefig(
                f'results/{dataset_name}_{attack_name}_sample000_{i}.png'
            )

            plt.close()

    # -------------------------
    # success rate 출력
    # -------------------------
    if total_samples > 0:
        success_rate = 100 * success_count / total_samples
    else:
        success_rate = 0

    print(
        f"{dataset_name} {attack_name} success rate: "
        f"{success_rate:.2f}%"
    )


# -------------------------------
# 6. 메인
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in ['MNIST', 'CIFAR10']:
        if dataset_name == 'MNIST':
            train_loader, test_loader = get_mnist_loaders()
            model = SimpleCNN(num_classes=10, input_channels=1).to(device)
            eps_values = {'FGSM':0.1, 'PGD':0.1}
        else:
            train_loader, test_loader = get_cifar10_loaders()
            model = SimpleCNN(num_classes=10, input_channels=3).to(device)
            eps_values = {'FGSM':0.1, 'PGD':0.1}

        print(f"=== Training {dataset_name} model ===")
        train(model, device, train_loader, epochs=1)
        print(f"=== Testing attacks on {dataset_name} ===")

        attacks = {
            'Targeted_FGSM': fgsm_targeted,
            'Untargeted_FGSM': fgsm_untargeted,
            'Targeted_PGD': pgd_targeted,
            'Untargeted_PGD': pgd_untargeted
        }

        for name, fn in attacks.items():
            eps = eps_values['FGSM'] if 'FGSM' in name else eps_values['PGD']
            test_attack(model, device, test_loader, fn, name, eps=eps, dataset_name=dataset_name)
