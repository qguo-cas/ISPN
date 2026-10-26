import torch.nn as nn
import  torch
import torch.nn.functional as F
# def downsample_tensor(input_tensor, scale_factor):
#     # 使用平均池化进行降采样
#     output_tensor = F.avg_pool2d(input_tensor, kernel_size=scale_factor)
#     #max pool
#     output_tensor = F.max_pool2d(input_tensor, kernel_size=scale_factor)
#     return output_tensor
class DownsampleModule(nn.Module):
    def __init__(self, scale_factor):
        super(DownsampleModule, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=scale_factor)
        # self.max_pool = nn.MaxPool2d(kernel_size=scale_factor)

    def forward(self, input_tensor):
        # 使用平均池化进行降采样
        avg_pooled = self.avg_pool(input_tensor)
        # 使用最大池化进行降采样
        # max_pooled = self.max_pool(input_tensor)
        return avg_pooled

class ResidualFeatureBlock(nn.Module):
    def __init__(self):
        super(ResidualFeatureBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv1_Res=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_Res = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv1_Fusion = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):

        Cursory_out=self.prelu(self.conv1(x))
        Cursory_out = self.prelu(self.conv2(Cursory_out))
        #1
        mid_out = Cursory_out
        RES_out=self.prelu(self.bn1(self.conv1_Res(mid_out)))
        RES_out = self.bn2(self.conv2_Res(RES_out))
        RES_out = torch.add(RES_out, mid_out)
        RES_out = self.prelu(RES_out)
        #2
        mid_out = RES_out
        RES_out = self.prelu(self.bn1(self.conv1_Res(mid_out)))
        RES_out = self.bn2(self.conv2_Res(RES_out))
        RES_out = torch.add(RES_out, mid_out)
        RES_out = self.prelu(RES_out)
        # 3
        mid_out = RES_out
        RES_out = self.prelu(self.bn1(self.conv1_Res(mid_out)))
        RES_out = self.bn2(self.conv2_Res(RES_out))
        RES_out = torch.add(RES_out, mid_out)
        RES_out = self.prelu(RES_out)

        out=self.prelu(self.conv1_Fusion(RES_out))
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        channel_weights = self.fc(avg_pool).view(x.size(0), -1, 1, 1)
        x = x * channel_weights
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_cat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(pool_cat))
        x = x * attention_map
        return x

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(ChannelSpatialAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.conv_shape=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu=nn.ReLU()
    def forward(self, x):
        x_channel = self.channel_attention(x)
        x_spatial = self.spatial_attention(x)
        x_combined = x_channel + x_spatial
        x_out=self.relu(self.conv_shape(x_combined))
        return x_out


class UpsampleModule(nn.Module):
    def __init__(self, scale_factor, mode='bicubic', align_corners=False):#bicubic/bilinear
        super(UpsampleModule, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        # 使用 interpolate 函数进行上采样
        x_upsampled = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x_upsampled

class SpatialFeatureBlock(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialFeatureBlock, self).__init__()
        self.PFE = ResidualFeatureBlock()
        self.spatial_attention = SpatialAttention(kernel_size)
        self.conv_shape = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        SPAE_1=self.PFE(x)
        SPAE_2=self.spatial_attention(SPAE_1)
        SPAE=self.relu(self.conv_shape(SPAE_2))
        return SPAE

class SpectralFeatureBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SpectralFeatureBlock, self).__init__()
        self.PFE = ResidualFeatureBlock()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.conv_shape = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        SPEE_1=self.PFE(x)
        SPEE_2=self.channel_attention(SPEE_1)
        SPEE=self.relu(self.conv_shape(SPEE_2))
        return SPEE

class JointFeatureBlock(nn.Module):
    def __init__(self):
        super(JointFeatureBlock, self).__init__()
        self.PFE = ResidualFeatureBlock()
        self.two_att=ChannelSpatialAttention(in_channels=32,reduction=16,kernel_size=7)
        self.conv_shape = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        SPAEE_1=self.PFE(x)
        SPAEE_2=self.two_att(SPAEE_1)
        SPAEE=self.relu(self.conv_shape(SPAEE_2))
        return SPAEE

def stack(pan, r=4):
    return torch.cat([pan for _ in range(r)], dim=1)

class SpectralProjection(nn.Module):
    def __init__(self, in_channel=8, retio=4):#gf4 wv8
        super(SpectralProjection, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.in_channel = in_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel * retio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel * retio, in_channel, bias=False),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        y = self.avg(x2).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = torch.sum(out, dim=1, keepdim=True)
        out = stack(out, r=self.in_channel)

        return out

class FusionNet(nn.Module):
    def __init__(self,pan_channels,ms_channels):
        super(FusionNet, self).__init__()
        self.pan_encoder = nn.Conv2d(in_channels=pan_channels, out_channels=32, kernel_size=3, padding=1)
        self.ms_encoder = nn.Conv2d(in_channels=ms_channels, out_channels=32, kernel_size=3, padding=1)
        self.prior_encoder = nn.Conv2d(in_channels=ms_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu=nn.ReLU()
        self.spem=SpectralFeatureBlock(in_channels=32)
        self.spam= SpatialFeatureBlock()
        self.spaem = JointFeatureBlock()
        self.conv_inject = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)
        self.conv_inspa=nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_constraint = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pan_decoder=nn.Conv2d(in_channels=32, out_channels=pan_channels, kernel_size=3, padding=1)
        self.ms_decoder = nn.Conv2d(in_channels=32, out_channels=ms_channels, kernel_size=3, padding=1)
        self.fusion_decoder=nn.Conv2d(in_channels=32, out_channels=ms_channels, kernel_size=3, padding=1)
        self.gray = SpectralProjection(in_channel=ms_channels)
    def forward(self, pan,ms,prior_img):
        #encoder
        pan_encode=self.relu(self.pan_encoder(pan))
        ms_encode = self.relu(self.ms_encoder(ms))
        prior_img_encode=self.relu(self.prior_encoder(prior_img))
        spea_detail=self.spaem(prior_img_encode)
        #1
        spa_detail1=self.spam(pan_encode)
        spe_detail1=self.spem(ms_encode)
        inject_spa1=torch.cat([spe_detail1,spa_detail1,ms_encode],dim=1)#96
        res_inject1=self.relu(self.conv_inject(inject_spa1))
        out_orispa1=torch.cat([spa_detail1,pan_encode],dim=1)#64
        out_spa1=self.relu(self.conv_inspa(out_orispa1))

        #2
        spa_detail2 = self.spam(out_spa1)
        spe_detail2 = self.spem(res_inject1)
        inject_spa2=torch.cat([spe_detail2,spa_detail2,ms_encode],dim=1)
        res_inject2 = self.relu(self.conv_inject(inject_spa2))
        consraint_1=torch.cat([res_inject2,spea_detail],dim=1)#64
        consraint_inspa1=self.relu(self.conv_constraint(consraint_1))
        out_orispa2 = torch.cat([spa_detail2, pan_encode], dim=1)  # 64
        out_spa2 = self.relu(self.conv_inspa(out_orispa2))

        #3
        spa_detail3 = self.spam(out_spa2)
        spe_detail3 = self.spem(consraint_inspa1)
        inject_spa3 = torch.cat([spe_detail3, spa_detail3, ms_encode], dim=1)
        res_inject3 = self.relu(self.conv_inject(inject_spa3))
        out_orispa3 = torch.cat([spa_detail3, pan_encode], dim=1)  # 64
        out_spa3 = self.relu(self.conv_inspa(out_orispa3))

        # 4
        spa_detail4 = self.spam(out_spa3)
        spe_detail4 = self.spem(res_inject3)
        inject_spa4 = torch.cat([spe_detail4, spa_detail4, ms_encode], dim=1)
        res_inject4 = self.relu(self.conv_inject(inject_spa4))
        out_orispa4 = torch.cat([spa_detail4, pan_encode], dim=1)  # 64
        out_spa4 = self.relu(self.conv_inspa(out_orispa4))

        # 5
        spa_detail5 = self.spam(out_spa4)
        spe_detail5 = self.spem(res_inject4)
        inject_spa5 = torch.cat([spe_detail5, spa_detail5, ms_encode], dim=1)
        res_inject5 = self.relu(self.conv_inject(inject_spa5))
        consraint_2 = torch.cat([res_inject5, spea_detail], dim=1)  # 64
        consraint_inspa2 = self.relu(self.conv_constraint(consraint_2))
        out_orispa5 = torch.cat([spa_detail5, pan_encode], dim=1)  # 64
        out_spa5 = self.relu(self.conv_inspa(out_orispa5))

        #6
        spa_detail6 = self.spam(out_spa5)
        spe_detail6= self.spem(consraint_inspa2)
        inject_spa6 = torch.cat([spe_detail6, spa_detail6, ms_encode], dim=1)
        res_inject6 = self.relu(self.conv_inject(inject_spa6))
        res=self.relu(torch.cat([res_inject6,pan_encode,ms_encode],dim=1))
        res_out = self.relu(self.conv_inject(res))
        out_orispa6 = torch.cat([spa_detail6, pan_encode], dim=1)  # 64
        out_spa6 = self.relu(self.conv_inspa(out_orispa6))
        lrms_up_gray = self.gray(ms)
        P= self.pan_decoder(out_spa6)
        M= self.ms_decoder(spe_detail6)
        out = self.fusion_decoder(res_out)

        return out,P,M,lrms_up_gray
