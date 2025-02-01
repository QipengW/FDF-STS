from sklearn.metrics.pairwise import cosine_similarity
node_number = 862
predict_len = 96
history_len = 96
class FDTS(nn.Module):
    def __init__(self):
        #in_c：输入维度  hid_c:隐藏层维度 out_c:输出维度
        super(GCN_F,self).__init__()
        self.node_embeddings = nn.Parameter(torch.rand(node_number,10))
        self.h1 = nn.Parameter(torch.randn(1, 16))
        self.z1 = nn.Parameter(torch.randn(1, 16))
        self.h2 = nn.Parameter(torch.randn(1, 16))
        self.z2 = nn.Parameter(torch.randn(1, 16))
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        self.h11 = nn.Parameter(torch.randn(16, 32))
        self.z11 = nn.Parameter(torch.randn(16, 32))
        self.h22 = nn.Parameter(torch.randn(16, 32))
        self.z22 = nn.Parameter(torch.randn(16, 32))
        
        self.Linear_3 = nn.Linear(1, 32)
        self.Linear_4 = nn.Linear(history_len*32, 16*history_len)
        self.Linear_5 = nn.Linear(16*history_len, history_len)
        self.act = nn.Tanh()
        

    def forward(self,data): 
        B,N,T = data["flow_x"].size(0),data["flow_x"].size(1),data["flow_x"].size(2)
        graph_data =  torch.tensor(cosine_similarity(self.node_embeddings.cpu().detach().numpy())).cuda()
        I = torch.eye(N, dtype=graph_data.dtype).to(device)
        D = torch.diag(torch.sum(graph_data, dim=-1) ** (-1 / 2)).to(device)
        L = torch.eye(graph_data.size(0),  dtype=graph_data.dtype).to(device) - torch.mm(torch.mm(D, graph_data), D)#[N,N]
        data_1 = torch.matmul(L,data["flow_x"]).unsqueeze(3)
        data_1 = self.Linear_3(data_1)#[B,N,T,32]
        #映射到频域
        out_1 = torch.fft.rfft(data["flow_x"], dim=2, norm='ortho')#[B,N,T,32]
        Low = out_1[:,:,0:(T//2+1)//5].unsqueeze(3)  #[B,N,(T//2+1)//10,32]
        High = out_1[:,:,(T//2+1)//5:].unsqueeze(3)  #[B,N,T-(T//2+1)//10,32]
        #低频特征提取
        x_low_real = self.act(torch.matmul(Low.real,self.h1)-torch.matmul(Low.imag,self.z1))
        x_low_imag = self.act(torch.matmul(Low.real,self.z1)+torch.matmul(Low.imag,self.h1))
        x_low_real_1 = torch.matmul(x_low_real,self.h11)-torch.matmul(x_low_imag,self.z11)
        x_low_imag_1 = torch.matmul(x_low_real,self.z11)+torch.matmul(x_low_imag,self.h11)
        x_low = torch.stack([x_low_real_1, x_low_imag_1], dim=-1) #[B,N,48,32,2]
        #高频特征提取
        x_high_real = self.act(torch.matmul(High.real,self.h2)-torch.matmul(High.imag,self.z2))
        x_high_imag = self.act(torch.matmul(High.real,self.z2)+torch.matmul(High.imag,self.h2))
        x_high_real_1 = torch.matmul(x_high_real,self.h22)-torch.matmul(x_high_imag,self.z22)
        x_high_imag_1 = torch.matmul(x_high_real,self.z22)+torch.matmul(x_high_imag,self.h22)
        x_high = torch.stack([x_high_real_1, x_high_imag_1], dim=-1) #[B,N,433,32,2]
        zero = torch.zeros(B,N,9,32,2).cuda()
        x_high = torch.cat((zero,x_high),dim=2)
        #将提取后的特征矩阵转回复数形式
        x_low = torch.view_as_complex(x_low)   #[B,N,48,32]
        x_high = torch.view_as_complex(x_high) #[B,N,433,32]
        #映射回时域
        x_low = torch.fft.irfft(x_low, n=96, dim=2, norm='ortho')   #[B,N,960,32]
        x_high = torch.fft.irfft(x_high, n=96, dim=2, norm='ortho') #[B,N,960,32]
        #解码，获得预测
        out_1 = x_high+data_1  #[B,N,T,64]
        out_2 = self.Linear_4(out_1.reshape(B,N,-1))
        out_3 = self.Linear_5(out_2)
        return out_3.squeeze()