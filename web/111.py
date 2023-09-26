import mindcv

model=mindcv.create_model('convnext_tiny',pretrained=True)
print(model)