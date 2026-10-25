"""Backbone registry for IC013. Wraps SOAP's get_models.get_dense_backbone (dinov2_base, mae_base,
capi_large unchanged) and adds ibot_base and dinov3_base, which SOAP's repo does not provide.
All return a model whose forward(x, vit_feat='out', vit_layer=-1) gives (B, N_patches, D), prefix tokens removed."""
import os, sys, torch, timm
from types import MethodType
SOAP_ROOT = os.environ.get('SOAP_ROOT', '/workspace/soap')
sys.path.insert(0, SOAP_ROOT)
import get_models as gm

IBOT_URL = 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth'

def _wrap(model, patch_size):
    model.patch_size = patch_size
    model.feat_dim = model.embed_dim
    model.forward = MethodType(gm.dinov2_forward, model)
    return model

def get_backbone(name):
    if name in ('dinov2_base', 'mae_base', 'capi_large'):
        return gm.get_dense_backbone(name)
    if name == 'dinov3_base':
        m = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True, dynamic_img_size=True)
        return _wrap(m, 16)
    if name == 'ibot_base':
        m = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True, num_classes=0)
        sd = torch.hub.load_state_dict_from_url(IBOT_URL, map_location='cpu')
        sd = sd.get('state_dict', sd.get('teacher', sd))
        sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith('head')}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing, f'iBOT missing keys: {missing}'
        assert all('masked_embed' in u for u in unexpected), f'iBOT unexpected keys: {unexpected}'
        return _wrap(m, 16)
    raise NotImplementedError(name)

PATCH = {'dinov2_base': 14, 'capi_large': 14, 'mae_base': 16, 'ibot_base': 16, 'dinov3_base': 16}
BACKBONES = ['dinov2_base', 'ibot_base', 'mae_base', 'capi_large', 'dinov3_base']
