import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import DigitDataset
from torch.utils.data import dataloader, random_split


class DigitCnn(nn.Module):
    def __init__(self, num_clases=46):
        super(DigitCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_clases),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    dataset = DigitDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCnn(num_clases=46).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    epochs = 100
    best_acc = 0
    model_name = "digit_cnn.pth"
    for epoch in range(epochs):
        # train loop
        model.train()
        total_loss = 0
        if epoch == 50:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # eval loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, prediction = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (prediction == labels).sum().item()
        val_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc >= best_acc:
            best_acc = val_acc

            print(f"best model saved - Acc = {best_acc:.2f}")
            torch.save(model.state_dict(), model_name)

    # Saving to onnx
    model.load_state_dict(torch.load(model_name))
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        f"{model_name[:-4]}.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_data"],
        output_names=["outputs_data"],
        dynamic_axes={
            "input_data": {0: "batch_size"},
            "output_data": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    train()
