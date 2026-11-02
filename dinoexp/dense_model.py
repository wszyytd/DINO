"""Reference-conditioned spatial encoder, multiview fusion and local/global Null."""
import torch
from torch import nn
from torch.nn import functional as F
from .local_attention import SpatialAdapter, CrossBlock, LocalMatcher


def fuse_spatial(patches, indices, weights, reliability, grid_size):
    """Bilinear native-grid fusion; coverage is independent of predicted probability."""
    b = patches.shape[0]
    if (indices.shape != (*patches.shape,4) or weights.shape != indices.shape
            or reliability.shape != patches.shape[:2]):
        raise ValueError('invalid dense fusion shapes')
    influence = weights * reliability[...,None,None]
    numer = torch.zeros(b,grid_size**2,device=patches.device,dtype=patches.dtype)
    cover = torch.zeros_like(numer)
    numer.scatter_add_(1,indices.reshape(b,-1), (patches[...,None]*influence).reshape(b,-1))
    cover.scatter_add_(1,indices.reshape(b,-1), influence.reshape(b,-1))
    average = numer / cover.clamp_min(1e-12)
    spatial = average / average.sum(1,keepdim=True).clamp_min(1e-12)
    return spatial, cover


class DenseSearchHead(nn.Module):
    def __init__(self,input_dim=768,dim=256,layers=4,heads=8,grid_size=32,reference_grid=8):
        super().__init__()
        if min(input_dim,dim,layers,heads,grid_size,reference_grid)<1 or dim%heads:
            raise ValueError('positive sizes and dimension divisible by heads required')
        self.config=dict(input_dim=input_dim,dim=dim,layers=layers,heads=heads,
                         grid_size=grid_size,reference_grid=reference_grid)
        self.grid_size=grid_size; self.reference_grid=reference_grid
        self.reference_adapter=SpatialAdapter(input_dim,dim,layers)
        self.view_adapter=SpatialAdapter(input_dim,dim,layers)
        self.reference_query=nn.Parameter(torch.randn(1,1,dim)*.02)
        self.reference_blocks=nn.ModuleList([CrossBlock(dim,heads) for _ in range(2)])
        self.geometry=nn.Sequential(nn.Linear(6,dim),nn.GELU(),nn.Linear(dim,dim))
        self.context=nn.TransformerEncoderLayer(dim,heads,dim*2,dropout=0.,
                                                batch_first=True,norm_first=True)
        self.global_context=nn.Linear(dim,dim)
        self.correlation_query=nn.Linear(dim,dim)
        self.correlation_reference=nn.Linear(dim,dim)
        self.correlation=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.view_state=nn.Sequential(nn.Linear(8,dim),nn.GELU(),nn.Linear(dim,dim))
        self.missing_state=nn.Parameter(torch.zeros(1,dim))
        self.match_blocks=nn.ModuleList([CrossBlock(dim,heads) for _ in range(2)])
        self.patch_score=nn.Linear(dim,1)
        self.local_null=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,1))
        self.reliability=nn.Sequential(nn.Linear(dim*3,dim),nn.GELU(),nn.Linear(dim,1))
        self.global_existence=nn.Sequential(nn.Linear(dim*2+2,dim),nn.GELU(),nn.Linear(dim,1))

    def freeze_real_branches(self):
        for module in (self.view_state,self.local_null,self.global_existence):
            module.requires_grad_(False)
        self.missing_state.requires_grad_(False)

    def _reference(self,x,valid):
        r=self.reference_adapter(x,valid)
        # Same validity-weighted 8x8 pooling contract as the retained baseline.
        r,keep=LocalMatcher.pool_reference(self,r,valid)
        description=self.reference_query.expand(len(r),-1,-1)
        for block in self.reference_blocks:
            description,_=block(description,r,keep)
        return r,keep,description[:,0]

    def forward(self,reference,reference_valid,views,view_valid,geometry,indices,weights,
                priors,view_weights,camera_state,camera_valid):
        b,v,l,n,c=views.shape
        valid=view_valid.bool()
        if not valid.any(-1).all() or (priors<=0).any() or (view_weights<=0).any():
            raise ValueError('every view needs valid content and positive fusion weights')
        r,rvalid,description=self._reference(reference,reference_valid.bool())
        q=self.view_adapter(views.reshape(b*v,l,n,c),valid.reshape(b*v,n)).reshape(b,v,n,-1)
        d=q.shape[-1]
        geo=self.geometry(geometry)
        q=q+geo[:,:,None,:]
        q=self.context(q.reshape(b*v,n,d),src_key_padding_mask=~valid.reshape(b*v,n)).reshape(b,v,n,d)
        full=(q[:,0]*valid[:,0,:,None]).sum(1)/valid[:,0].sum(1,keepdim=True)
        # Exact full-image duplicate tiles retain the same representation as full.
        local=(geometry[:,:,:4]-geometry[:,0:1,:4]).abs().sum(-1)>1e-7
        q=q+self.global_context(full)[:,None,None,:]*local[:,:,None,None]
        projected_q=F.normalize(self.correlation_query(q),dim=-1)
        projected_r=F.normalize(self.correlation_reference(r),dim=-1)
        similarity=torch.einsum('bvnd,bmd->bvnm',projected_q,projected_r)
        similarity=similarity.masked_fill(~rvalid[:,None,None,:],-torch.inf)
        maximum=similarity.max(-1).values
        smooth=torch.logsumexp(similarity,-1)-rvalid.sum(-1).log()[:,None,None]
        overall=(projected_q*F.normalize(self.correlation_reference(description),dim=-1)[:,None,None]).sum(-1)
        state=torch.where(camera_valid[:,None].bool(),self.view_state(camera_state),self.missing_state)
        q=q+self.correlation(torch.stack((maximum,smooth,overall),-1))+state[:,None,None]
        memory=r[:,None].expand(-1,v,-1,-1).reshape(b*v,r.shape[1],d)
        memory_valid=rvalid[:,None].expand(-1,v,-1).reshape(b*v,-1)
        q=q.reshape(b*v,n,d)
        for block in self.match_blocks:
            q,_=block(q,memory,memory_valid)
        q=q.reshape(b,v,n,d)
        summaries=(q*valid[:,:,:,None]).sum(2)/valid.sum(2,keepdim=True)
        desc=description[:,None].expand(-1,v,-1)
        null_logits=self.local_null(torch.cat((desc,summaries),-1)).squeeze(-1)
        patch_logits=self.patch_score(q).squeeze(-1).masked_fill(~valid,-torch.inf)
        per_view=torch.cat((patch_logits,null_logits[:,:,None]),-1).softmax(-1)
        # Keep a conditional spatial distribution even when Null saturates in float32.
        floor=torch.cat((valid,torch.ones(b,v,1,device=valid.device,dtype=torch.bool)),-1)
        per_view=per_view.clamp_min(1e-12)*floor
        per_view=per_view/per_view.sum(-1,keepdim=True)
        reliability=self.reliability(torch.cat((desc,summaries,geo),-1)).squeeze(-1).sigmoid()
        effective=(reliability+1e-6)*priors*view_weights
        spatial,coverage=fuse_spatial(per_view[:,:,:-1],indices,weights*valid[:,:,:,None],effective,self.grid_size)
        if not (coverage.sum(1)>0).all():
            raise ValueError('no native search-grid coverage')
        normalized=view_weights/view_weights.sum(1,keepdim=True)
        summary=(summaries*normalized[:,:,None]).sum(1)
        local_null=per_view[:,:,-1]
        stats=torch.stack(((local_null*normalized).sum(1),local_null.max(1).values),-1)
        presence=self.global_existence(torch.cat((description,summary,stats),-1)).sigmoid()
        probabilities=torch.cat((spatial*presence,1-presence),-1)
        return dict(probabilities=probabilities,spatial=probabilities[:,:-1].reshape(b,self.grid_size,self.grid_size),
                    null=probabilities[:,-1],local_null=local_null,coverage=coverage,
                    reliability=reliability,conditioned_features=q,reference_tokens=r,
                    reference_descriptor=description)
