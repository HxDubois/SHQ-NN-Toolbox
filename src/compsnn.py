import torch
from torch import nn

class Temp2SpaceNN(nn.Module):
    """
    Temporal to Spatial representation neural network

    Attributes
    ----------
    temporal: torch.nn.Module
        Temporal feature extraction component
    spatial: torch.nn.Module
        Spatial feature extraction component

    """
    def __init__(self, n_nodes, out_size) -> None:
        """
        Initializes the Temp2SpaceNN.

        Parameters
        ----------
        n_nodes: int
            number of spatial nodes
        out_size: int
            dimension of the outputted embedding
        """
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(8, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.spatial = nn.Sequential(
            nn.Linear(n_nodes * 64, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, out_size),
        )

    def pooling(self, x, b) -> torch.Tensor:
        """
        Applies a pooling operation projecting temporal data on a graph

        Parameters
        ----------
        x: torch.Tensor
            temporal data signal. shape (batch size,length,number of features)
        b: torch.Tensor
            graph contribution matrix.:math:`b_{nij}` defines the contribution for sample:math:`n` of time index:math:`i` to node:math:`j`. shape(batch size,length,number of nodes)

        Returns
        ----------
        z: torch.Tensor
            spatial signal. shape (batch size,number of features, number of nodes)
        """
        y = torch.einsum("nij,nik->nijk", b, x)
        z, _ = torch.max(y, axis=1)
        return z

    def forward(self, x, b) -> torch.Tensor:
        """
        Processes a batch of samples.

        Parameters
        ----------
        x: torch.Tensor
            temporal data of shape (batch size, length, 8)
        b: torch.Tensor
            node assignation matrix of shape (batch size, length, number of nodes)
        
        Returns
        ----------
        z: torch.Tensor
            embedding of shape (batch size,:attr:`out_size`)
        """
        n = x.shape[0]
        x = torch.swapaxes(x, 1, 2)
        x = self.temporal(x)
        x = torch.swapaxes(x, 1, 2)
        x = self.pooling(x, b).reshape(n, -1)
        return self.spatial(x)    
    
class HGNN(nn.Module):
    """
    Hierarchical Graph Neural Network
    
    

    Attributes
    ----------
    tree: torch.Tensor
        Graph adjacency matrix

    """
    def __init__(self, tree, in_size=16, out_size=10) -> None:
        """
        Initializes the HGNN.

        Parameters
        ----------
        in_size: int
            signal dimension on leaves
        out_size: int
            dimension of the outputted embedding
        """
        super().__init__()
        self.tree = torch.Tensor(tree)
        self.f_micro = nn.Sequential(
            nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
        )

        self.f_macro = nn.Sequential(nn.Linear(64, 16), nn.ReLU())

        self.f_out = nn.Sequential(
            nn.Linear(tree.shape[0] * 16, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, out_size),
        )

    def forward(self, x):
        """
        Processes a batch of samples.

        Parameters
        ----------
        x: torch.Tensor
            signal of shape (batch size, number of leaves,:attr:`in_size`)
        
        Returns
        ----------
        z: torch.Tensor
            embedding of shape (batch size,:attr:`out_size`)
        """
        x = self.f_micro(x)
        h = torch.zeros(size=(x.shape[0], self.tree.shape[0], 64))
        for i in range(self.tree.shape[0]):
            h[:, i,:], _ = torch.multiply(torch.swapaxes(x, 1, 2), self.tree[i]).max(
                axis=2
            )
        h = self.f_macro(h).reshape(-1, 16 * self.tree.shape[0])
        return self.f_out(h)
    
class MultiScaleConv1d(nn.Module):
    def __init__(self, hi_in, lo_in, hi_out, lo_out, k_hi=3, k_lo=3, d_lo=2) -> None:
        super().__init__()
        self.hi_conv = nn.Conv1d(hi_in, hi_out, k_hi, padding=k_hi // 2)
        self.lo_conv = nn.Conv1d(
            lo_in, lo_out, k_lo, dilation=d_lo, padding=(d_lo * (k_lo - 1)) // 2
        )

    def forward(self, x):
        x_hi = self.hi_conv(x)
        x_lo = self.lo_conv(x)
        return torch.cat([x_hi, x_lo], dim=1)
    
class TempNN(nn.Module):
    """
    Temporal Neural Network with multiscale convolutions
    
    

    Attributes
    ----------
    conv: torch.nn.Module
        Convolution component
    out: torch.nn.Module
        Fully-connected embedding component

    """
    def __init__(self, layers, h=128, out_size=16, l=300) -> None:
        """
        Initializes the Temporal NN.

        Parameters
        ----------
        layers: List[int]
            signal dimension at each convolution layer
        h: int
            number of hidden neurons in the:attr:`out` network
        out_size: int
            dimension of the outputted embedding
        l: int
            length of the temporal data
        """
        super().__init__()
        f = []
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            hi_out = l_out // 2
            lo_out = l_out - hi_out
            f.append(MultiScaleConv1d(l_in, hi_out, 5, l_in, lo_out, 7, 2))
            f.append(nn.MaxPool1d(2))
            f.append(nn.BatchNorm1d(l_out))
            f.append(nn.ReLU())
        self.conv = nn.Sequential(*f)

        self.out = nn.Sequential(
            nn.Linear((l // (2 ** (len(layers) - 1))) * layers[-1], h),
            nn.ReLU(),
            nn.BatchNorm1d(h),
            nn.Linear(h, out_size),
        )

    def forward(self, x):
        """
        Processes a batch of samples.

        Parameters
        ----------
        x: torch.Tensor
            signal of shape (batch size,:attr:`l`,:attr:`layers[0]`)
        
        Returns
        ----------
        z: torch.Tensor
            embedding of shape (batch size,:attr:`out_size`)
        """
        x = torch.swapaxes(x, 1, 2)
        n = x.shape[0]
        x = self.conv(x)
        return self.out(x.reshape(n, -1))
    
class CompSNN(nn.Module):
    """
    Composite Signal Neural Network. It analyses trajectory data by using representations learned both from a spatial or a temporal perspective.    

    Attributes
    ----------
    t2snn: Temp2SpaceNN
        Temporo-spatial component: it analyses the signal from a temporal point of view using 1d convolutions, then projects it onto a spatial representation from which it produces an embedding.
    hgnn: HGNN
        Spatial component: it processes spatially defined signal on the nodes of a tree-structured graph, producing the signal aggregated on the root node as an embedding.
    cnn: TempNN
        Temporal component: it processes temporally defined signal using multi-scale convolutions. Features learned by the convolutionnal part are used by a fully connected layer to produce an embedding.
    f: torch.nn.Module
        Aggregator: it aggregates the representations produced by each component to output a final representation which can be used for a specific task (ie. it can be a class probability prediction vector).

    """
    def __init__(
        self,
        n_nodes,
        adj,
        h=[64, 64, 64],
        input_size=16,
        graph_size=100,
        output_size=16,
    ) -> None:
        """
        Initializes the CompSNN.

        Parameters
        ----------
        n_nodes: int
            number of spatial nodes
        adj: torch.Tensor
            adjacency matrix of the HGNN
        h: List[int]
            dimensions of the outputted embeddings for each component
        input_size: int
            dimension of the temporal signal
        gaph_size: int
            number of dimensions of the signal on the HGNN leaves
        output_size: int
            size of the representation produced by the CompSNN
        """
        super().__init__()
        self.t2snn = Temp2SpaceNN(n_nodes=n_nodes, out_size=h[0])
        self.hgnn = HGNN(tree=adj, in_size=graph_size, out_size=h[1])
        self.cnn = TempNN([input_size, 64, 32], out_size=h[2])

        self.f = nn.Sequential(
            nn.Linear(sum(h), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x, b, t):
        """
        Processes a batch of samples.

        Parameters
        ----------
        x: torch.Tensor
            graph signal
        b: torch.Tensor
            spatial contribution matrix
        t: torch.Tensor
            temporal signal
        
        Returns
        ----------
        z: torch.Tensor
            embedding of shape (batch size,:attr:`output_size`)
        """
        z_tg = self.t2snn(t, b)
        z_g = self.hgnn(x)
        z_t = self.cnn(t)
        z = self.f(torch.cat([z_tg, z_g, z_t], dim=1))
        return z