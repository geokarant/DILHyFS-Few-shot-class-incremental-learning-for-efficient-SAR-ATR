import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinear
from backbone.resnet import resnet18
from backbone.gfnet import GFNetPyramid
from functools import partial
from copy import deepcopy

def get_backbone(args, backbone_type, pretrained=False):
    name = backbone_type.lower()
    if name ==  'gfnet-h-ti':
        model = GFNetPyramid(
                img_size=args["input_dim"], 
                patch_size=4, embed_dim=[64, 128, 256, 512], depth=[3, 3, 10, 3],
                mlp_ratio=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,
            )
        checkpoint = torch.load("./weights/gfnet-h-ti.pth")
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'aux_head.weight', 'aux_head.bias']:
            if k in checkpoint_model:
               print(f"Removing key {k} from pretrained checkpoint")
               del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)
        model.out_dim=512
        return model
    elif name == "resnet18":
        model = resnet18(pretrained=pretrained,args=args).to('cuda')
        model.out_dim = 512 
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(args, args["backbone_type"], pretrained)
        self.fc = None
        self._device = args["device"][0]
        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim
        
    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)
    
    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.W_rand = None
        self.RP_dim = None
        self.backbone_type = args["backbone_type"]
        self._cur_task = 0
        self.args= args
    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def enable_parameters(self):
    # Freeze all parameters
        for name, param in self.named_parameters():
            # Enable training for SSF parameters
            if "ssf" in name:
                param.requires_grad = True
            # Enable training for specific fully connected layers
            elif 'fc.weight' in name or 'fc.sigma' in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.args['backbone_ft']
    def extract_vector(self, x):
        if "resnet" in self.backbone_type:
            x= self.backbone(x)['features']
        elif self.backbone_type == 'clip':
            x= self.backbone.encode_image(x)
        else:
            x= self.backbone(x)
        return x
    def forward(self, x):
        x = self.extract_vector(x)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)  
        out = self.fc(x)
        out.update({"features": x})
        return out

class Cross_Attention(nn.Module):
    def __init__(self, args, pretrained):
        super(Cross_Attention, self).__init__()
        # for RanPAC
        self.args = args
        self.W_rand = None
        self.RP_dim = None
        self.anchors = None
        self._cur_task = 0           
        self.backbone_type_vit = self.args["backbone_type_vit"]
        self.backbone_type_cnn = self.args["backbone_type_cnn"]
        self.backbone_vit = get_backbone(self.args, self.backbone_type_vit, pretrained= True)
        self.backbone_cnn = get_backbone(self.args, self.backbone_type_cnn, pretrained= True)
        self.tuning_mode = args["tuning_mode"]
        self.dim_0= 64
        if 'resnet' not in self.backbone_type_vit :
            self.vit_patch_embed = deepcopy(self.backbone_vit.patch_embed)
            self.vit_layers = nn.ModuleList([
                deepcopy(self.backbone_vit.layers[i]) for i in range(4)
            ])
            if 'tiny' not in self.args['backbone_type_vit']:
                self.vit_norm= deepcopy(self.backbone_vit.norm)
                self.vit_avg_pool= deepcopy(self.backbone_vit.avgpool)
        else:
            self.vit_layers = nn.ModuleList([
                deepcopy(self.backbone_vit.layer1),  deepcopy(self.backbone_vit.layer2),  deepcopy(self.backbone_vit.layer3),  deepcopy(self.backbone_vit.layer4)
            ])
            self.vit_patch_embed = deepcopy(self.backbone_vit.patch_embed)
            self.vit_avg_pool= self.backbone_vit.avgpool
        self.cnn_patch_embeds = nn.ModuleList([
            deepcopy(self.backbone_cnn.patch_embed[i]) for i in range(4)
        ])
        self.cnn_blocks = nn.ModuleList([
            deepcopy(self.backbone_cnn.blocks[i]) for i in range(4)
        ])
        self.cnn_norm = deepcopy(self.backbone_cnn.norm)
        self.cnn_pos_embed = deepcopy(self.backbone_cnn.pos_embed)  # Copy pos_embed
        # Delete the original backbones to save memory
        del self.backbone_vit
        del self.backbone_cnn
               
        self.block_dims = [self.dim_0, 2*self.dim_0, 4*self.dim_0, 8*self.dim_0]
        if self.tuning_mode == 'ssf':
        # Initialize SSF scale and shift parameters for each block
            self.ssf_scales = nn.ParameterList()
            self.ssf_shifts = nn.ParameterList()
            for dim in  self.block_dims:  # One set of scale and shift per block with specific dimensions
                scale, shift = self.init_ssf_scale_shift(dim)
                self.ssf_scales.append(scale)
                self.ssf_shifts.append(shift)
                #self.attention_layer = attention_layer(emb_dim=args["d_model"],
                #    tf_layers= args["tf_layers"], tf_head= args["tf_head"], tf_dim=args["ff_dim"])
        self.fc = None
        self._device = args["device"][0]
        self.enable_parameters()
    @property
    def feature_dim(self):
        return self.args["d_model"]
    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    def enable_parameters(self):
    # Freeze all parameters
        for name, param in self.named_parameters():
            # Enable training for SSF parameters
            if "ssf" in name:
                param.requires_grad = True
            # Enable training for specific fully connected layers
            elif 'fc.weight' in name or 'fc.sigma' in name:
                param.requires_grad = True
            # Set training status based on the backbone_ft flag for all other layers
            elif 'vit' in name:
                param.requires_grad = self.args['backbone_vit_ft']
            elif 'cnn' in name:
                param.requires_grad = self.args['backbone_cnn_ft']

    def fusion_add (self, x1,x2):
        x = x1 + x2
        #print(x.shape,x1.shape,x2.shape)
        #x= torch.cat([x1,x2] , dim=0)
        return x

        
    def init_ssf_scale_shift(self,dim):
        scale = nn.Parameter(torch.ones(dim))
        shift = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)

        return scale, shift

    def ssf_ada(self, x, scale, shift):
        assert scale.shape == shift.shape
        if x.shape[-1] == scale.shape[0]:
            return x * scale + shift
        elif x.shape[1] == scale.shape[0]:
            return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
        else:
            raise ValueError('the input tensor shape does not match the shape of the scale factor.')
    def process_vit(self, x_vit, block_dim, reshape_dim=None):
            if 'tiny' in self.backbone_type_vit or 'resnet' in self.backbone_type_vit:
                if reshape_dim:
                    x_vit = x_vit.reshape(x_vit.shape[0],reshape_dim, reshape_dim, block_dim)  # [B, H*W, C] → [B, H, W, C]
                x_vit = x_vit.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
            return x_vit
    def reshape_vit(self,x_vit, block_dim, spatial_dim):
            if 'tiny' in self.backbone_type_vit or 'resnet' in self.backbone_type_vit:
                return x_vit.permute(0, 2, 3, 1).reshape(x_vit.shape[0], spatial_dim * spatial_dim, block_dim)
            return x_vit
    def extract_vector(self, x_vit, x_cnn):
        # First Block
        for i in range(4):  # Iterate over the four blocks
            if i == 0:
                x_cnn = self.cnn_patch_embeds[i](x_cnn)
                x_cnn = self.cnn_blocks[i](x_cnn)
                x_vit = self.vit_patch_embed(x_vit)
                x_vit = self.vit_layers[i](x_vit)
            else:
                x_cnn = self.cnn_patch_embeds[i](x)
                x_cnn = self.cnn_blocks[i](x_cnn)

                x_vit = self.process_vit(x, self.block_dims[i-1], reshape_dim=56 // (2**(i-1)))
                x_vit = self.vit_layers[i](x_vit)
            
            x_vit = self.reshape_vit(x_vit, self.block_dims[i], spatial_dim= 56 // (2**i))
            x_cnn = x_cnn + (self.cnn_pos_embed if i == 0 else 0)
            if i == 3:
                x_vit = self.vit_avg_pool(x_vit)
                x_vit = torch.flatten(x_vit, 1)
                x_cnn = self.cnn_norm(x_cnn).mean(1)
            x = self.fusion_add(x_vit, x_cnn)
            if self.tuning_mode == 'ssf':
                x = self.ssf_ada(x, self.ssf_scales[i], self.ssf_shifts[i])  # Apply SSF for Block i
        return x
    def forward(self, x_vit, x_cnn):
        x = self.extract_vector(x_vit,x_cnn)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        #print("x before fc", x.shape)
        out = self.fc(x)
        out.update({"features": x})
        return out
    



