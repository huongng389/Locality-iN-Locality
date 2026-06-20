# Huong dan cai tien mo hinh GTSRB

## Model de xuat

File model: `GTSRB_StrongModel.py`

Mo hinh dung backbone ConvNeXt-Tiny pretrained ImageNet-22K/1K tu `timm`, them adapter 192 chieu va classification head 43 lop. Adapter 192 chieu giup model van tuong thich voi notebook goc:

```python
from GTSRB_StrongModel import LNL_Ti as small
model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)
```

Mo hinh tu normalize input theo ImageNet mean/std trong `forward`, nen tap test cua notebook goc chi can `Resize` va `ToTensor`.

## Train checkpoint

Chay cac cell tai du lieu GTSRB trong `Instructions.ipynb` truoc, sau do:

```bash
cd /content/Locality-iN-Locality
pip install timm
python train_gtsrb_strong.py --data-root ./data --epochs 30 --batch-size 64 --amp --imagenet-pretrained
```

Checkpoint tot nhat duoc luu tai:

```text
pretrained_gtsrb_strong.pth
```

## Load checkpoint trong notebook

```python
from GTSRB_StrongModel import LNL_Ti as small

model = small(pretrained=False)
ckpt = torch.load("pretrained_gtsrb_strong.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model = model.cuda()
```

Sau do dung cell test goc:

```python
model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print("Standard accuracy: %.2f %%" % (100 * float(correct) / total))
```

## Ly do cai tien

- Backbone ConvNeXt-Tiny pretrained manh hon TNT-Tiny train tu dau trong notebook goc.
- Fine-tune voi AdamW, cosine learning rate, label smoothing va augmentation phu hop bien bao giao thong.
- Input normalization duoc dong goi trong model de tranh sai khac giua train va test notebook.
- Head 192 chieu giu tinh plug-and-play voi cau truc notebook goc.
