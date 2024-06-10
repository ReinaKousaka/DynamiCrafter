import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
from lvdm.basics import zero_module
from collections import namedtuple
from functools import wraps

import torch
import torch.nn.functional as F
from packaging import version
from torch import einsum, nn

# constants

AttentionConfig = namedtuple(
    "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)

# helpers


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=True):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask = mask, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v, mask)

        scale = q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=4, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, coords, rays=None):
        embed_fns = []
        num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * self.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers
        sines = torch.sin(scaled_coords).reshape(num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape( num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                 relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77,mult = None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        
        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)) )
        if mult and image_cross_attention_scale_learnable:
            self.cc_proj = self.cc_proj = nn.Linear(query_dim +6*mult*mult, inner_dim)
            self.act = nn.Sigmoid()


    def forward(self, x, context=None, mask=None, cam = None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)


        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)


        if out_ip is not None:
            if self.image_cross_attention_scale_learnable and cam is not None:
                beta_z = self.act(self.cc_proj(torch.cat([x,cam],dim=-1)))
                out = out + self.image_cross_attention_scale * out_ip * beta_z
            else:
                out = out + self.image_cross_attention_scale * out_ip
        
        return self.to_out(out)
    
    def efficient_forward(self, x, context=None, mask=None, cam = None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        
        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable and cam is not None:
                beta_z = self.act(self.cc_proj(torch.cat([x,cam],dim=-1)))
                out = out + self.image_cross_attention_scale * out_ip * beta_z
            else:
                out = out + self.image_cross_attention_scale * out_ip
           
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77,mult = None):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, video_length=video_length, image_cross_attention=image_cross_attention, image_cross_attention_scale=image_cross_attention_scale, image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,text_context_len=text_context_len,mult = mult)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if mult and image_cross_attention_scale_learnable:
            self.pixel_shuffle = nn.PixelUnshuffle(mult)


    def forward(self, x, context=None, mask=None, **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            if self.image_cross_attention_scale_learnable and kwargs:
                plucker_embedding = self.pixel_shuffle(rearrange(kwargs['plucker_embedding'], 'b t c h w -> (b t) c h w').contiguous())
                plucker_embedding = rearrange(plucker_embedding, 'b c h w -> b (h w) c')
                input_tuple = (x, context, plucker_embedding)

            else:
                input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)


    def _forward(self, x, context=None, plucker_embedding = None, mask=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask, cam = plucker_embedding) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, video_length=None,
                 image_cross_attention=False, image_cross_attention_scale_learnable=False,mult = None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                image_cross_attention=image_cross_attention,
                image_cross_attention_scale_learnable=True,
                mult = mult,
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self.mult = mult



    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False, causal_block_size=1,
                 relative_position=False, temporal_length=None,mult = 0, index =0):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self.cc_proj = nn.Linear(inner_dim + 96 +6*mult*mult, inner_dim)
        self.pose_encode = PositionalEncoding()
        self.mult = mult
        self.index = index
        self.pixel_shuffle = nn.PixelUnshuffle(mult)

    def forward(self, x, context=None, **kwargs):

        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)


        if kwargs:
            plucker_embedding = self.pixel_shuffle(rearrange(kwargs['plucker_embedding'], 'b t c h w -> (b t) c h w').contiguous())
            plucker_embedding = rearrange(plucker_embedding, '(b t) c h w -> (b h w) t c', b =b)
            camera_embed = kwargs['camera_embeddings']

            camera_embed = camera_embed.repeat_interleave(repeats=w * h, dim=0)
            camera_embed = rearrange(camera_embed, 'b t c-> (b t) c')
            camera_embed = self.pose_encode(camera_embed)
            camera_embed = rearrange(camera_embed, '(b t) c -> b t c', t=t)
            x = self.cc_proj(torch.cat([x,camera_embed,plucker_embedding],dim=-1))

        temp_mask = None
        if self.causal_attention:
            # slice the from mask map
            temp_mask = self.mask[:,:t,:t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        else:
            mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in
    

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_



class CrossAttention_Epipolar(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False,
                 image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.relative_position = relative_position
        if self.relative_position:
            assert (temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)))

    def forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale  # TODO check
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask < 0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2)  # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip = torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)

        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        h = self.heads
        if exists(mask):
            # min_val_fp16 = torch.finfo(torch.float16).min
            mask[mask == 0] = -1000
            mask[mask != -1000] = 0
            mask = mask.repeat_interleave(h,0).to(q.dtype)
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask, op=None)

        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        # if exists(mask):
        #     raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        return self.to_out(out)

class BasicTransformerBlock_Epipolar(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
        super().__init__()
        # attn_cls = CrossAttention_Epipolar if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention_Epipolar(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=None)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint


    # def forward(self, x, context=None, mask=None, **kwargs):
    #     ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
    #     input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
    #     if context is not None:
    #         input_tuple = (x, context,mask)
    #     if mask is not None:
    #         forward_mask = partial(self._forward, mask=mask)
    #         return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
    #     return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)


    def forward(self, x, context=None, mask=None, **kwargs):
        x = self.attn2(self.norm2(x), context=None) + x
        x = self.attn1(self.norm1(x), context= None, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x

class EpipolarTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=True, video_length=None,
                 image_cross_attention=False, image_cross_attention_scale_learnable=False, mult = 0, index = 0):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.mult = mult
        self.index = index

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock_Epipolar(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear


    def forward(self, x, context=None, mask = None, **kwargs):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)

        x = rearrange(x, 'b c t h w-> b (t h w) c').contiguous()
        x = self.proj_in(x)

        # for s in range(3):
        #     tmp = kwargs['epipolar_masks'][s].shape
        #     print(f'shape {s}: {tmp}')
        if self.mult == 8:
            mask = kwargs['epipolar_masks'][2]
        elif self.mult == 4:
            mask = kwargs['epipolar_masks'][1]
        elif self.mult == 2:
            mask = kwargs['epipolar_masks'][0]

        for i, block in enumerate(self.transformer_blocks):
            if mask is not None:
                x = block(x, context=context, mask = mask.half(), **kwargs)
            else:
                x = block(x, context=context, **kwargs)

        x = self.proj_out(x)
        x = rearrange(x, 'b (t h w) c -> b c t h w', b=b,  h=h, w=w).contiguous()

        return x + x_in


# class CrossAttention_Epipolar(nn.Module):

#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
#                  relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False,
#                  image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         self.dim_head = dim_head
#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.attend = Attend(flash=True)

#         self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

#         self.relative_position = relative_position
#         if self.relative_position:
#             assert (temporal_length is not None)
#             self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
#             self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
#         else:
#             ## only used for spatial attention, while NOT for temporal attention
#             if XFORMERS_IS_AVAILBLE and temporal_length is None:
#                 self.forward = self.forward

#         self.video_length = video_length
#         self.image_cross_attention = image_cross_attention
#         self.image_cross_attention_scale = image_cross_attention_scale
#         self.text_context_len = text_context_len
#         self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
#         if self.image_cross_attention:
#             self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
#             self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
#             if image_cross_attention_scale_learnable:
#                 self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)))

#     def forward(self, x, context=None, mask=None):
#         spatial_self_attn = (context is None)
#         k_ip, v_ip, out_ip = None, None, None

#         h = self.heads
#         q = self.to_q(x)
#         context = default(context, x)

#         if self.image_cross_attention and not spatial_self_attn:
#             context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
#             k = self.to_k(context)
#             v = self.to_v(context)
#             k_ip = self.to_k_ip(context_image)
#             v_ip = self.to_v_ip(context_image)
#         else:
#             if not spatial_self_attn:
#                 context = context[:, :self.text_context_len, :]
#             k = self.to_k(context)
#             v = self.to_v(context)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
#         out = self.attend(q,k,v,mask=mask)
#         # sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
#         # if self.relative_position:
#         #     len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
#         #     k2 = self.relative_position_k(len_q, len_k)
#         #     sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale  # TODO check
#         #     sim += sim2
#         # del k
#         #
#         # if exists(mask):
#         #     ## feasible for causal attention mask only
#         #     max_neg_value = -torch.finfo(sim.dtype).max
#         #     mask = repeat(mask, 'b i j -> (b h) i j', h=h)
#         #     sim.masked_fill_(~(mask > 0.5), max_neg_value)
#         #
#         # # attention, what we cannot get enough of
#         # sim = sim.softmax(dim=-1)
#         # out = torch.einsum('b i j, b j d -> b i d', sim, v)
#         # if self.relative_position:
#         #     v2 = self.relative_position_v(len_q, len_v)
#         #     out2 = einsum('b t s, t s d -> b t d', sim, v2)  # TODO check
#         #     out += out2
#         out = rearrange(out, 'b h n d -> b n (h d)', h=h)

#         ## for image cross-attention
#         if k_ip is not None:
#             k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
#             sim_ip = torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
#             del k_ip
#             sim_ip = sim_ip.softmax(dim=-1)
#             out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
#             out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)

#         if out_ip is not None:
#             if self.image_cross_attention_scale_learnable:
#                 out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
#             else:
#                 out = out + self.image_cross_attention_scale * out_ip

#         return self.to_out(out)

#     def efficient_forward(self, x, context=None, mask=None):
#         spatial_self_attn = (context is None)
#         k_ip, v_ip, out_ip = None, None, None

#         q = self.to_q(x)
#         context = default(context, x)

#         if self.image_cross_attention and not spatial_self_attn:
#             context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
#             k = self.to_k(context)
#             v = self.to_v(context)
#             k_ip = self.to_k_ip(context_image)
#             v_ip = self.to_v_ip(context_image)
#         else:
#             if not spatial_self_attn:
#                 context = context[:, :self.text_context_len, :]
#             k = self.to_k(context)
#             v = self.to_v(context)

#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.unsqueeze(3)
#             .reshape(b, t.shape[1], self.heads, self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b * self.heads, t.shape[1], self.dim_head)
#             .contiguous(),
#             (q, k, v),
#         )
#         # actually compute the attention, what we cannot get enough of
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

#         ## for image cross-attention
#         if k_ip is not None:
#             k_ip, v_ip = map(
#                 lambda t: t.unsqueeze(3)
#                 .reshape(b, t.shape[1], self.heads, self.dim_head)
#                 .permute(0, 2, 1, 3)
#                 .reshape(b * self.heads, t.shape[1], self.dim_head)
#                 .contiguous(),
#                 (k_ip, v_ip),
#             )
#             out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
#             out_ip = (
#                 out_ip.unsqueeze(0)
#                 .reshape(b, self.heads, out.shape[1], self.dim_head)
#                 .permute(0, 2, 1, 3)
#                 .reshape(b, out.shape[1], self.heads * self.dim_head)
#             )

#         if exists(mask):
#             raise NotImplementedError
#         out = (
#             out.unsqueeze(0)
#             .reshape(b, self.heads, out.shape[1], self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b, out.shape[1], self.heads * self.dim_head)
#         )
#         if out_ip is not None:
#             if self.image_cross_attention_scale_learnable:
#                 out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
#             else:
#                 out = out + self.image_cross_attention_scale * out_ip

#         return self.to_out(out)
# class BasicTransformerBlock_Epipolar(nn.Module):

#     def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
#                 disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
#         super().__init__()
#         # attn_cls = CrossAttention_Epipolar if attention_cls is None else attention_cls
#         self.disable_self_attn = disable_self_attn
#         self.attn1 = CrossAttention_Epipolar(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
#             context_dim=None)
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
#         self.attn2 = CrossAttention_Epipolar(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
#             context_dim=None)
#         self.image_cross_attention = image_cross_attention

#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.checkpoint = checkpoint


#     # def forward(self, x, context=None, mask=None, **kwargs):
#     #     ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
#     #     input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
#     #     if context is not None:
#     #         input_tuple = (x, context,mask)
#     #     if mask is not None:
#     #         forward_mask = partial(self._forward, mask=mask)
#     #         return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
#     #     return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)


#     def forward(self, x, context=None, mask=None, **kwargs):
#         x = self.attn1(self.norm1(x), context= None, mask=mask) + x
#         x = self.attn2(self.norm2(x), context=None) + x
#         x = self.ff(self.norm3(x)) + x
#         return x

# class EpipolarTransformer(nn.Module):
#     """
#     Transformer block for image-like data in spatial axis.
#     First, project the input (aka embedding)
#     and reshape to b, t, d.
#     Then apply standard transformer action.
#     Finally, reshape to image
#     NEW: use_linear for more efficiency instead of the 1x1 convs
#     """

#     def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
#                  use_checkpoint=True, disable_self_attn=False, use_linear=True, video_length=None,
#                  image_cross_attention=False, image_cross_attention_scale_learnable=False):
#         super().__init__()
#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
#         self.proj_in = nn.Linear(in_channels, inner_dim)

#         attention_cls = None
#         self.transformer_blocks = nn.ModuleList([
#             BasicTransformerBlock_Epipolar(
#                 inner_dim,
#                 n_heads,
#                 d_head,
#                 dropout=dropout,
#                 context_dim=context_dim,
#                 attention_cls=attention_cls,
#                 checkpoint=use_checkpoint) for d in range(depth)
#         ])
#         self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
#         self.use_linear = use_linear


#     def forward(self, x, context=None, mask = None, **kwargs):
#         b, c, t, h, w = x.shape
#         x_in = x
#         x = self.norm(x)

#         x = rearrange(x, 'b c t h w-> b (t h w) c').contiguous()
#         x = self.proj_in(x)

#         for i, block in enumerate(self.transformer_blocks):
#             if mask is not None:
#                 x = block(x, context=context, mask = mask, **kwargs)
#             else:
#                 x = block(x, context=context, **kwargs)

#         x = self.proj_out(x)
#         x = rearrange(x, 'b (t h w) c -> b c t h w', b=b,  h=h, w=w).contiguous()

#         return x + x_in