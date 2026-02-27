from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, Union, Any


class MVDNetConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    # --- 基础参数 (必须和网络底层的变量名一模一样) ---
    mid: int = 64                 
    num_blocks: int = 15
    lookahead: int = 2
    propagation_order: int = 1    
    scale: int = 2                
    
    # --- 激活函数 ---
    feat_act_cfg: Union[str, Dict[str, Any]] = "LeakyReLU"
    body_act_cfg: Union[str, Dict[str, Any]] = "PReLU"
    
    # ==========================================
    # --- 子模块组件配置 (字典占位，供 registry 反射) ---
    # ==========================================
    feature_extractor_cfg: Dict[str, Any] = {'type': 'ResidualFeatureExtractor', 'in_channels': 4} 
    mv_refiner_cfg: Dict[str, Any] = {'type': 'LiteConvexMVRefiner'}
    gmc_cfg: Dict[str, Any] = {'type': 'GMC'}
    
    residual_block_cfg: Dict[str, Any] = {'type': 'StandardResidualBlock'}
    
    # 网络需要的三种融合模块
    foward_backward_fusion_cfg: Dict[str, Any] = {'type': 'BasicConcatFusion'}
    h_prop_current_feat_fusion_cfg: Dict[str, Any] = {'type': 'SoftGateAdaptiveFusion'}
    first_2nd_order_fusion_cfg: Dict[str, Any] = {'type': 'DeinterlacingMultiOrderFusion'}
    
    deint_up_cfg: Dict[str, Any] = {'type': 'DeintShufflePack', 'scale_factor': 2}