import torch, math
from thop import profile, clever_format
from spcnn import SPCNN

if __name__=="__main__":
    custom_ops = {}
    input = torch.randn(1, 3, 224, 224)


    # model = SPCNN(dims=[32,64,128,256], layers=[3,6,9,3], mlp_ratio=3.0, expand_ratio=1.0, drop_path_rate=0.00)
    # model = SPCNN(dims=[40,80,160,320], layers=[3,6,12,3], mlp_ratio=3.0, expand_ratio=1.0, drop_path_rate=0.05)
    # model = SPCNN(dims=[48,96,192,384], layers=[4,7,15,4], mlp_ratio=3.0, expand_ratio=1.0, drop_path_rate=0.10)
    # model = SPCNN(dims=[64,128,256,512], layers=[4,7,15,4], mlp_ratio=3.0, expand_ratio=1.0, drop_path_rate=0.20)
    # model = SPCNN(dims=[96,192,384,768], layers=[5,9,17,5], mlp_ratio=2.0, expand_ratio=1.0, drop_path_rate=0.35)

    model = SPCNN(dims=[64,128,256,512], layers=[4,7,15,4], mlp_ratio=3.0, expand_ratio=1.0, drop_path_rate=0.20)
    model = SPCNN(dims=[64,128,256,512], layers=[3,6,12,3], mlp_ratio=4.0, expand_ratio=1.0, drop_path_rate=0.20)
    model = SPCNN(dims=[64,128,256,512], layers=[5,9,23,5], mlp_ratio=2.0, expand_ratio=1.0, drop_path_rate=0.20)

    model = SPCNN(dims=[64,128,256,512], layers=[3,6,14,3], mlp_ratio=3.0, expand_ratio=2.0, drop_path_rate=0.20)
    model = SPCNN(dims=[64,128,256,512], layers=[3,6,11,3], mlp_ratio=3.0, expand_ratio=3.0, drop_path_rate=0.20)

    model.eval()
    print(model)
    
    macs, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print('Flops:  ', macs)
    print('Params: ', params)

