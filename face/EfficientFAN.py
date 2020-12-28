import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from efficientnet_custom import EfficientNet as Efficientnet_custom
from models.gcnet import ContextBlock
from models.sagan import Self_Attn
import torch.nn.functional as F

# teacher
# EfficinetNet-b7 - output : [nb, 640, 8, 8]
# modules[0]: 1st deconv - output : [nb, 256, 16, 16]
# modules[1]: 2st deconv - output : [nb, 256, 32, 32]
# modules[2]: 3st deconv - output : [nb, 256, 64, 64]
# modules[3]: regression layer(1x1conv)

# student
# EfficinetNet-b0 - output : [nb, 320, 8, 8]
# modules[0]: 1st deconv - output : [nb, 128, 16, 16]
# modules[1]: 2st deconv - output : [nb, 128, 32, 32]
# modules[2]: 3st deconv - output : [nb, 128, 64, 64]
# modules[3]: regression layer(1x1conv)

class get_affinity_graph_matrix(nn.Module):
    def __init__(self, n):
        super(get_affinity_graph_matrix, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)  # self로 선언
        # self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)  # self로 선언
        self.n = n

    def forward(self, input):
        '''
        output : affinity graph : list형.
        affinity_graph[k] : 2**k x 2**k 사이즈의 패치로 노드화 시킨 feature map을 affinity graph로 만든것.
        즉 k가 작을수록 matrix가 크다.
        '''
        nb = input.shape[0]
        # affinity_graph = torch.zeros(nb, self.n, 4 ** self.n, 4 ** self.n)
        input_save = input

        affinity_graph = []
        # k = 0,1,2,...,n-1
        # 하나의 patch size에 대해 실행
        for k in range(self.n):
            if k != 0:
                input = self.pooling(input)  # node 크기 확장, shape : [channel, H/2, W/2]
            elif k == 0:
                input = input_save

            # print(f'input shape : {input.shape}')

            # vectorize along channel
            A_vec_T = torch.reshape(input, (input.shape[0], input.shape[1], -1))  # [B, C , HxW]
            A_vec = A_vec_T.transpose(1, 2)  # [B, H * W , C], vectorization

            num = torch.bmm(A_vec, A_vec_T)  # [B, HxW, HxW]
            # print(num.shape)
            assert (num.shape == (nb, input.shape[2] * input.shape[3], input.shape[2] * input.shape[3]))

            # make denominator
            A_norm = torch.unsqueeze(torch.norm(A_vec, dim=2), 2) + 1e-10

            '''
            affinity_graph라는 리스트에 3차원 텐서저장.
            텐서의 평면은 affinity_graph를 의미하고 채널축은 batch를 의미한다.
            즉, shape : (batch, 4 ** (self.n - k), 4 ** (self.n - k))
            '''
            affinity_graph.append(torch.div(torch.div(num, A_norm).transpose(1, 2), A_norm).transpose(1,2)) # 하나의 patch에 대한 affinity graph matrix

            assert (affinity_graph[k].shape == (nb, 4 ** (self.n - k), 4 ** (self.n - k)))

        assert (len(affinity_graph) == self.n)
        return affinity_graph

class Swish(nn.Module):
    def __init__(self):
        '''
         Examples:
            >>> m = Swish()
            >>> input = torch.randn(2)
            >>> output = m(input)
        '''
        super(Swish,self).__init__()

    def forward(self,x):
        return x * torch.sigmoid(x)

class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

class Teacher(nn.Module):
    def __init__(self,num_joints):
        super(Teacher, self).__init__()

        self.num_joints = num_joints

        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b7 = EfficientNet.from_name('efficientnet-b7')

        modules = nn.ModuleList([])
        # Add 1st deconv block (img size = )
        modules.append(nn.Sequential(nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))


        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(nn.Conv2d(256, num_joints, kernel_size=1))

        self.module = nn.ModuleList(modules)
        self.bn = nn.BatchNorm2d(num_joints)
        self.swish = Swish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

        self.PS3 = get_affinity_graph_matrix(n=3)
        self.PS4 = get_affinity_graph_matrix(n=4)
        self.PS5 = get_affinity_graph_matrix(n=5)
        # self.PS6 = get_affinity_graph_matrix(n=6)

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)  # self로 선언

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        outputs[2,3,4,5] : PS matrix
        outputs[6,7,8,9] : FA map
        '''
        # EfficientNet output = [C, H, W] = [batch, 640, 8, 8]
        B1 = self.efficientnet_b7(x) # [256,8,8]
        B2 = self.module[0](B1)  # dconv : [256,16,16]
        B3 = self.module[1](B2)  # dconv : [256,32,32]
        B4 = self.module[2](B3)  # dconv : [256,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        H = self.bn(H)
        H = self.swish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        # # Patch Similarity Distillation
        # AG1 = self.PS3(B1)  # [nb,64,64], [nb,16,16], [nb,4,4]
        # AG2 = self.PS4(B2)  # [nb,256,256], ...
        # AG3 = self.PS5(B3)  # [nb,1024,1024], ...
        # PS6 = get_affinity_graph_matrix(n=6)
        # AG4 = PS6(B4)

        return [H, coord, B1,B2,B3,B4]
        # return [H, coord]

class Student(nn.Module):
    def __init__(self, num_joints):
        super(Student, self).__init__()

        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b0 = EfficientNet.from_name('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 16 -> 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 32 -> 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(nn.Conv2d(128, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        # self.hswish = nn.Hardswish()
        self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

        # 1x1 conv : feature aligned map
        self.conv1x1_1 = nn.Conv2d(320, 640, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(128, 256, kernel_size=1)

        self.PS3 = get_affinity_graph_matrix(n=3)
        self.PS4 = get_affinity_graph_matrix(n=4)
        self.PS5 = get_affinity_graph_matrix(n=5)
        self.PS6 = get_affinity_graph_matrix(n=6)

        # ps6을 넣을 memory가 딸려서 B4는 pooling 해서 ps5 에 넣는다.
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)  # self로 선언

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        outputs[2,3,4,5] : PS matrix
        outputs[6,7,8,9] : FA map
        '''
        # EfficientNet output = [C, H, W] = [batch, 320, 8, 8]
        B1 = self.efficientnet_b0(x)  # [320,8,8]
        B2 = self.module[0](B1)  # dconv : [128,16,16]
        B3 = self.module[1](B2)  # dconv : [128,32,32]
        B4 = self.module[2](B3)  # dconv : [128,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        H = self.bn(H)
        H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap) # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        '''AG1 = self.PS3(B1)  # [nb,64,64], [nb,16,16], [nb,4,4]
        AG2 = self.PS4(B2)  # [nb,256,256], ...
        AG3 = self.PS5(B3)  # [nb,1024,1024], ...

        PS6 = get_affinity_graph_matrix(n=6)
        AG4 = PS6(B4)'''

        # B1,B2,B3,B4 나가기전에 CONV거쳐야해
        B1_conv = self.conv1x1_1(B1) # [640,8,8]
        B2_conv = self.conv1x1_2(B2) # [256,16,16]
        B3_conv = self.conv1x1_3(B3) # [256,32,32]
        B4_conv = self.conv1x1_4(B4) # [256,64,64]

        # return [H, coord, B1, B2, B3, B4]
        # return [H, coord, B1, B2, B3, B4]
        return [H, coord, B1_conv, B2_conv, B3_conv, B4_conv]
        # return [H, coord]

class Teacher_pretrained(nn.Module):
    def __init__(self,num_joints):
        super(Teacher_pretrained, self).__init__()

        self.num_joints = num_joints

        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b7 = EfficientNet.from_pretrained('efficientnet-b7')

        modules = nn.ModuleList([])
        # Add 1st deconv block (img size = )
        modules.append(nn.Sequential(nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))


        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(nn.Conv2d(256, num_joints, kernel_size=1))

        self.module = nn.ModuleList(modules)
        self.bn = nn.BatchNorm2d(num_joints)
        # self.swish = Swish()
        self.swish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        outputs[2,3,4,5] : PS matrix
        outputs[6,7,8,9] : FA map
        '''
        # EfficientNet output = [C, H, W] = [batch, 640, 8, 8]
        B1 = self.efficientnet_b7(x) # [256,8,8]
        B2 = self.module[0](B1)  # dconv : [256,16,16]
        B3 = self.module[1](B2)  # dconv : [256,32,32]
        B4 = self.module[2](B3)  # dconv : [256,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        # H = self.bn(H)
        # H = self.swish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

class Teacher_pretrained_2(nn.Module):
    def __init__(self,num_joints):
        super(Teacher_pretrained_2, self).__init__()

        self.num_joints = num_joints

        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b7 = EfficientNet.from_pretrained('efficientnet-b7')

        modules = nn.ModuleList([])
        # Add 1st deconv block (img size = )
        modules.append(nn.Sequential(nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))


        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(nn.Conv2d(256, num_joints, kernel_size=1))

        self.module = nn.ModuleList(modules)
        self.bn = nn.BatchNorm2d(num_joints)
        self.swish = Swish()
        # self.swish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        outputs[2,3,4,5] : PS matrix
        outputs[6,7,8,9] : FA map
        '''
        # EfficientNet output = [C, H, W] = [batch, 640, 8, 8]
        B1 = self.efficientnet_b7(x) # [256,8,8]
        B2 = self.module[0](B1)  # dconv : [256,16,16]
        B3 = self.module[1](B2)  # dconv : [256,32,32]
        B4 = self.module[2](B3)  # dconv : [256,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        H = self.bn(H)
        H = self.swish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

class Student_pretrained(nn.Module):
    def __init__(self, num_joints):
        super(Student_pretrained, self).__init__()

        # self.flag = flag
        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b0 = EfficientNet.from_pretrained('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 16 -> 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 32 -> 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(
            nn.Conv2d(128, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        self.hswish = nn.Hardswish()
        # self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        '''
        # EfficientNet output = [C, H, W] = [batch, 320, 8, 8]
        B1 = self.efficientnet_b0(x)  # [320,8,8]
        B2 = self.module[0](B1)  # dconv : [128,16,16]
        B3 = self.module[1](B2)  # dconv : [128,32,32]
        B4 = self.module[2](B3)  # dconv : [128,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        H = self.bn(H)
        H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

class Student_enc_custom(nn.Module):
    def __init__(self, num_joints):
        super(Student_enc_custom, self).__init__()

        # self.flag = flag
        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b0 = Efficientnet_custom.from_name('efficientnet-b0')
        # self.efficientnet_b0 = EfficientNet.from_name('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 16 -> 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 32 -> 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(
            nn.Conv2d(128, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        # self.hswish = nn.Hardswish()
        self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        '''
        # EfficientNet output = [C, H, W] = [batch, 320, 8, 8]
        B1 = self.efficientnet_b0(x)  # [320,8,8]
        B2 = self.module[0](B1)  # dconv : [128,16,16]
        B3 = self.module[1](B2)  # dconv : [128,32,32]
        B4 = self.module[2](B3)  # dconv : [128,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        H = self.bn(H)
        H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

class Teacher_enc_custom(nn.Module):
    def __init__(self, num_joints):
        super(Teacher_enc_custom, self).__init__()

        # self.flag = flag
        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b7 = Efficientnet_custom.from_name('efficientnet-b7')
        # self.efficientnet_b0 = EfficientNet.from_name('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(
            nn.Conv2d(256, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        # self.swish = Swish()
        self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        outputs[2,3,4,5] : PS matrix
        outputs[6,7,8,9] : FA map
        '''
        # EfficientNet output = [C, H, W] = [batch, 640, 8, 8]
        B1 = self.efficientnet_b7(x)  # [256,8,8]
        B2 = self.module[0](B1)  # dconv : [256,16,16]
        B3 = self.module[1](B2)  # dconv : [256,32,32]
        B4 = self.module[2](B3)  # dconv : [256,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        H = self.bn(H)
        H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

class Student_pretrained_2(nn.Module):
    def __init__(self, num_joints):
        super(Student_pretrained_2, self).__init__()

        # self.flag = flag
        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b0 = EfficientNet.from_pretrained('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 16 -> 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 32 -> 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(
            nn.Conv2d(128, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        # self.hswish = nn.Hardswish()
        # self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        '''
        # EfficientNet output = [C, H, W] = [batch, 320, 8, 8]
        B1 = self.efficientnet_b0(x)  # [320,8,8]
        B2 = self.module[0](B1)  # dconv : [128,16,16]
        B3 = self.module[1](B2)  # dconv : [128,32,32]
        B4 = self.module[2](B3)  # dconv : [128,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        # H = self.bn(H)
        # H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

# relu in softargmax --> bn+hswish(my code)
class Student_pretrained_3(nn.Module):
    def __init__(self, num_joints):
        super(Student_pretrained_3, self).__init__()

        # self.flag = flag
        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b0 = EfficientNet.from_pretrained('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 16 -> 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 32 -> 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(
            nn.Conv2d(128, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        # self.hswish = nn.Hardswish()
        self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        '''
        # EfficientNet output = [C, H, W] = [batch, 320, 8, 8]
        B1 = self.efficientnet_b0(x)  # [320,8,8]
        B2 = self.module[0](B1)  # dconv : [128,16,16]
        B3 = self.module[1](B2)  # dconv : [128,32,32]
        B4 = self.module[2](B3)  # dconv : [128,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        # H = self.bn(H)
        # H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        hmap = self.bn(hmap)
        hmap = self.hswish(hmap)  # For numerical stability
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        # hmap = self.relu(hmap)
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

# relu in softargmax --> bn+hswish(nn.Hardswish)
class Student_pretrained_4(nn.Module):
    def __init__(self, num_joints):
        super(Student_pretrained_4, self).__init__()

        # self.flag = flag
        self.num_joints = num_joints
        # load EfficientNet without 1x1 Conv, pooling
        self.efficientnet_b0 = EfficientNet.from_pretrained('efficientnet-b0')
        modules = nn.ModuleList([])

        # Add 1st deconv block (img size = 8 -> 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 16 -> 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 32 -> 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True)))

        # Add regression layer (1x1 conv) ==> output : heatmap
        modules.append(
            nn.Conv2d(128, num_joints, kernel_size=1))  # 이게 그냥 1x1을 사용한걸까 아니면 EfficientNet같은 1x1을 사용한 걸까?

        self.module = nn.ModuleList(modules)

        # additional layer after 1x1 Conv before Softargmax
        # EfficientFAN 에서 이렇게 했다고 언급함.
        self.bn = nn.BatchNorm2d(num_joints)
        self.hswish = nn.Hardswish()
        # self.hswish = Hardswish()

        # For integration (softargmax layer)
        # heatmap size : 64*64, input size : 256*256
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

    def forward(self, x):
        '''
        outputs[0] : hmap
        outputs[1] : coordi
        '''
        # EfficientNet output = [C, H, W] = [batch, 320, 8, 8]
        B1 = self.efficientnet_b0(x)  # [320,8,8]
        B2 = self.module[0](B1)  # dconv : [128,16,16]
        B3 = self.module[1](B2)  # dconv : [128,32,32]
        B4 = self.module[2](B3)  # dconv : [128,64,64]
        H = self.module[3](B4)  # 1x1 regression layer
        # H = self.bn(H)
        # H = self.hswish(H)

        # hmap = torch.tensor(H)
        hmap = H.clone()
        hmap = self.bn(hmap)
        hmap = self.hswish(hmap)  # For numerical stability
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64 * 64))
        # hmap = self.relu(hmap)
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]
