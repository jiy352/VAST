

# RUN_MODEL_ode_system

# system evaluation block

# op _00Lo_linear_combination_eval
# LANG: u --> _00Lp
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1656__00Lp = -1*v1632_u

# op _00Lr_linear_combination_eval
# LANG: w --> _00Ls
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1657__00Ls = -1*v1668_w

# op _00Lq_indexed_passthrough_eval
# LANG: _00Lp, _00Ls --> frame_vel
# SHAPES: (1, 1), (1, 1) --> (1, 3)
# full namespace: adapter_comp
v2097_frame_vel__temp[i_v1656__00Lp__00Lq_indexed_passthrough_eval] = v1656__00Lp.flatten()
v2097_frame_vel = v2097_frame_vel__temp.copy()
v2097_frame_vel__temp[i_v1657__00Ls__00Lq_indexed_passthrough_eval] = v1657__00Ls.flatten()
v2097_frame_vel = v2097_frame_vel__temp.copy()

# op _00LL_decompose_eval
# LANG: frame_vel --> _00LQ, _00LM
# SHAPES: (1, 3) --> (1, 1), (1, 1)
# full namespace: MeshPreprocessing_comp
v1670__00LM = ((v2097_frame_vel.flatten())[src_indices__00LM__00LL]).reshape((1, 1))
v1672__00LQ = ((v2097_frame_vel.flatten())[src_indices__00LQ__00LL]).reshape((1, 1))

# op _00LN_linear_combination_eval
# LANG: _00LM --> _00LO
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v1671__00LO = -1*v1670__00LM

# op _00LR_linear_combination_eval
# LANG: _00LQ --> _00LS
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v1673__00LS = -1*v1672__00LQ

# op _00LP_indexed_passthrough_eval
# LANG: _00LO, _00LS, w --> fs
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v1669_fs__temp[i_v1671__00LO__00LP_indexed_passthrough_eval] = v1671__00LO.flatten()
v1669_fs = v1669_fs__temp.copy()
v1669_fs__temp[i_v1673__00LS__00LP_indexed_passthrough_eval] = v1673__00LS.flatten()
v1669_fs = v1669_fs__temp.copy()
v1669_fs__temp[i_v1668_w__00LP_indexed_passthrough_eval] = v1668_w.flatten()
v1669_fs = v1669_fs__temp.copy()

# op _00Ly_decompose_eval
# LANG: eel --> _00M8, _00Lz, _00LC, _00LZ, _00M1, _00M2, _00M7, _00Mq, _00Mr, _00MX, _00M_, _00N4
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 5, 3), (1, 40, 5, 3), (1, 1, 5, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 41, 4, 3), (1, 41, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1662__00Lz = ((v1785_eel.flatten())[src_indices__00Lz__00Ly]).reshape((1, 40, 5, 3))
v1664__00LC = ((v1785_eel.flatten())[src_indices__00LC__00Ly]).reshape((1, 40, 5, 3))
v1677__00LZ = ((v1785_eel.flatten())[src_indices__00LZ__00Ly]).reshape((1, 1, 5, 3))
v1679__00M1 = ((v1785_eel.flatten())[src_indices__00M1__00Ly]).reshape((1, 40, 4, 3))
v1680__00M2 = ((v1785_eel.flatten())[src_indices__00M2__00Ly]).reshape((1, 40, 4, 3))
v1683__00M7 = ((v1785_eel.flatten())[src_indices__00M7__00Ly]).reshape((1, 40, 4, 3))
v1684__00M8 = ((v1785_eel.flatten())[src_indices__00M8__00Ly]).reshape((1, 40, 4, 3))
v1694__00Mq = ((v1785_eel.flatten())[src_indices__00Mq__00Ly]).reshape((1, 41, 4, 3))
v1695__00Mr = ((v1785_eel.flatten())[src_indices__00Mr__00Ly]).reshape((1, 41, 4, 3))
v1714__00MX = ((v1785_eel.flatten())[src_indices__00MX__00Ly]).reshape((1, 40, 4, 3))
v1716__00M_ = ((v1785_eel.flatten())[src_indices__00M___00Ly]).reshape((1, 40, 4, 3))
v1719__00N4 = ((v1785_eel.flatten())[src_indices__00N4__00Ly]).reshape((1, 40, 4, 3))

# op _00Oj_decompose_eval
# LANG: eel_wake_coords --> _00Ok
# SHAPES: (1, 69, 5, 3) --> (1, 1, 5, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v1763__00Ok = ((v1761_eel_wake_coords.flatten())[src_indices__00Ok__00Oj]).reshape((1, 1, 5, 3))

# op _00LT_power_combination_eval
# LANG: fs --> _00LU
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v1674__00LU = (v1669_fs)
v1674__00LU = (v1674__00LU*_00LT_coeff).reshape((1, 3))

# op _00M3_linear_combination_eval
# LANG: _00M1, _00M2 --> _00M4
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1681__00M4 = v1679__00M1+v1680__00M2

# op _00M9_linear_combination_eval
# LANG: _00M8, _00M7 --> _00Ma
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1685__00Ma = v1683__00M7+v1684__00M8

# op _00Ol_power_combination_eval
# LANG: _00Ok --> _00Om
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v1764__00Om = (v1763__00Ok)
v1764__00Om = v1764__00Om.reshape((1, 1, 5, 3))

# op _00LV_power_combination_eval
# LANG: _00LU --> _00LW
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v1675__00LW = (v1674__00LU)
v1675__00LW = (v1675__00LW*_00LV_coeff).reshape((1, 3))

# op _00M5_power_combination_eval
# LANG: _00M4 --> _00M6
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1682__00M6 = (v1681__00M4)
v1682__00M6 = (v1682__00M6*_00M5_coeff).reshape((1, 40, 4, 3))

# op _00Mb_power_combination_eval
# LANG: _00Ma --> _00Mc
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1686__00Mc = (v1685__00Ma)
v1686__00Mc = (v1686__00Mc*_00Mb_coeff).reshape((1, 40, 4, 3))

# op _00Oi_indexed_passthrough_eval
# LANG: _00Om, eel_wake_coords --> eel_TE_wake_coords
# SHAPES: (1, 1, 5, 3), (1, 69, 5, 3) --> (1, 70, 5, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v1809_eel_TE_wake_coords__temp[i_v1761_eel_wake_coords__00Oi_indexed_passthrough_eval] = v1761_eel_wake_coords.flatten()
v1809_eel_TE_wake_coords = v1809_eel_TE_wake_coords__temp.copy()
v1809_eel_TE_wake_coords__temp[i_v1764__00Om__00Oi_indexed_passthrough_eval] = v1764__00Om.flatten()
v1809_eel_TE_wake_coords = v1809_eel_TE_wake_coords__temp.copy()

# op _00LA_power_combination_eval
# LANG: _00Lz --> _00LB
# SHAPES: (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v1663__00LB = (v1662__00Lz)
v1663__00LB = (v1663__00LB*_00LA_coeff).reshape((1, 40, 5, 3))

# op _00LD_power_combination_eval
# LANG: _00LC --> _00LE
# SHAPES: (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v1665__00LE = (v1664__00LC)
v1665__00LE = (v1665__00LE*_00LD_coeff).reshape((1, 40, 5, 3))

# op _00LX expand_array_eval
# LANG: _00LW --> _00LY
# SHAPES: (1, 3) --> (1, 1, 5, 3)
# full namespace: MeshPreprocessing_comp
v1676__00LY = np.einsum('ad,bc->abcd', v1675__00LW.reshape((1, 3)) ,np.ones((1, 5))).reshape((1, 1, 5, 3))

# op _00Md_linear_combination_eval
# LANG: _00M6, _00Mc --> eel_coll_pts_coords
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1987_eel_coll_pts_coords = v1682__00M6+v1686__00Mc

# op _00Pz_decompose_eval
# LANG: eel_TE_wake_coords --> _00PA, _00PB, _00PC, _00PD
# SHAPES: (1, 70, 5, 3) --> (1, 69, 4, 3), (1, 69, 4, 3), (1, 69, 4, 3), (1, 69, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1810__00PA = ((v1809_eel_TE_wake_coords.flatten())[src_indices__00PA__00Pz]).reshape((1, 69, 4, 3))
v1811__00PB = ((v1809_eel_TE_wake_coords.flatten())[src_indices__00PB__00Pz]).reshape((1, 69, 4, 3))
v1812__00PC = ((v1809_eel_TE_wake_coords.flatten())[src_indices__00PC__00Pz]).reshape((1, 69, 4, 3))
v1813__00PD = ((v1809_eel_TE_wake_coords.flatten())[src_indices__00PD__00Pz]).reshape((1, 69, 4, 3))

# op _00LF_linear_combination_eval
# LANG: _00LB, _00LE --> _00LG
# SHAPES: (1, 40, 5, 3), (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v1666__00LG = v1663__00LB+v1665__00LE

# op _00L__linear_combination_eval
# LANG: _00LZ, _00LY --> _00M0
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: MeshPreprocessing_comp
v1678__00M0 = v1677__00LZ+v1676__00LY

# op _00PE reshape_eval
# LANG: eel_coll_pts_coords --> _00PF
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1814__00PF = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00PK reshape_eval
# LANG: _00PA --> _00PL
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1817__00PL = v1810__00PA.reshape((1, 276, 3))

# op _00PY reshape_eval
# LANG: eel_coll_pts_coords --> _00PZ
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1824__00PZ = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00Q3 reshape_eval
# LANG: _00PB --> _00Q4
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1827__00Q4 = v1811__00PB.reshape((1, 276, 3))

# op _00Qh reshape_eval
# LANG: eel_coll_pts_coords --> _00Qi
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1834__00Qi = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00Qn reshape_eval
# LANG: _00PC --> _00Qo
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1837__00Qo = v1812__00PC.reshape((1, 276, 3))

# op _00LH_indexed_passthrough_eval
# LANG: _00LG, _00M0 --> eel_bd_vtx_coords
# SHAPES: (1, 40, 5, 3), (1, 1, 5, 3) --> (1, 41, 5, 3)
# full namespace: MeshPreprocessing_comp
v1988_eel_bd_vtx_coords__temp[i_v1666__00LG__00LH_indexed_passthrough_eval] = v1666__00LG.flatten()
v1988_eel_bd_vtx_coords = v1988_eel_bd_vtx_coords__temp.copy()
v1988_eel_bd_vtx_coords__temp[i_v1678__00M0__00LH_indexed_passthrough_eval] = v1678__00M0.flatten()
v1988_eel_bd_vtx_coords = v1988_eel_bd_vtx_coords__temp.copy()

# op _00PG expand_array_eval
# LANG: _00PF --> _00PH
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1815__00PH = np.einsum('abd,c->abcd', v1814__00PF.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00PM expand_array_eval
# LANG: _00PL --> _00PN
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1818__00PN = np.einsum('acd,b->abcd', v1817__00PL.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _00P_ expand_array_eval
# LANG: _00PZ --> _00Q0
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1825__00Q0 = np.einsum('abd,c->abcd', v1824__00PZ.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00Q5 expand_array_eval
# LANG: _00Q4 --> _00Q6
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1828__00Q6 = np.einsum('acd,b->abcd', v1827__00Q4.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _00QB reshape_eval
# LANG: eel_coll_pts_coords --> _00QC
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1844__00QC = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00QH reshape_eval
# LANG: _00PD --> _00QI
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1847__00QI = v1813__00PD.reshape((1, 276, 3))

# op _00Qj expand_array_eval
# LANG: _00Qi --> _00Qk
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1835__00Qk = np.einsum('abd,c->abcd', v1834__00Qi.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00Qp expand_array_eval
# LANG: _00Qo --> _00Qq
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1838__00Qq = np.einsum('acd,b->abcd', v1837__00Qo.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _00PI reshape_eval
# LANG: _00PH --> _00PJ
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1816__00PJ = v1815__00PH.reshape((1, 44160, 3))

# op _00PO reshape_eval
# LANG: _00PN --> _00PP
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1819__00PP = v1818__00PN.reshape((1, 44160, 3))

# op _00Q1 reshape_eval
# LANG: _00Q0 --> _00Q2
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1826__00Q2 = v1825__00Q0.reshape((1, 44160, 3))

# op _00Q7 reshape_eval
# LANG: _00Q6 --> _00Q8
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1829__00Q8 = v1828__00Q6.reshape((1, 44160, 3))

# op _00QD expand_array_eval
# LANG: _00QC --> _00QE
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1845__00QE = np.einsum('abd,c->abcd', v1844__00QC.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00QJ expand_array_eval
# LANG: _00QI --> _00QK
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1848__00QK = np.einsum('acd,b->abcd', v1847__00QI.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _00Ql reshape_eval
# LANG: _00Qk --> _00Qm
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1836__00Qm = v1835__00Qk.reshape((1, 44160, 3))

# op _00Qr reshape_eval
# LANG: _00Qq --> _00Qs
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1839__00Qs = v1838__00Qq.reshape((1, 44160, 3))

# op _00V6_decompose_eval
# LANG: eel_bd_vtx_coords --> _00V7, _00V8, _00V9, _00Va
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1989__00V7 = ((v1988_eel_bd_vtx_coords.flatten())[src_indices__00V7__00V6]).reshape((1, 40, 4, 3))
v1990__00V8 = ((v1988_eel_bd_vtx_coords.flatten())[src_indices__00V8__00V6]).reshape((1, 40, 4, 3))
v1991__00V9 = ((v1988_eel_bd_vtx_coords.flatten())[src_indices__00V9__00V6]).reshape((1, 40, 4, 3))
v1992__00Va = ((v1988_eel_bd_vtx_coords.flatten())[src_indices__00Va__00V6]).reshape((1, 40, 4, 3))

# op _00PQ_linear_combination_eval
# LANG: _00PJ, _00PP --> _00PR
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1820__00PR = v1816__00PJ+-1*v1819__00PP

# op _00Q9_linear_combination_eval
# LANG: _00Q2, _00Q8 --> _00Qa
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1830__00Qa = v1826__00Q2+-1*v1829__00Q8

# op _00QF reshape_eval
# LANG: _00QE --> _00QG
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1846__00QG = v1845__00QE.reshape((1, 44160, 3))

# op _00QL reshape_eval
# LANG: _00QK --> _00QM
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1849__00QM = v1848__00QK.reshape((1, 44160, 3))

# op _00Qt_linear_combination_eval
# LANG: _00Qm, _00Qs --> _00Qu
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1840__00Qu = v1836__00Qm+-1*v1839__00Qs

# op _00VB reshape_eval
# LANG: _00V8 --> _00VC
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2006__00VC = v1990__00V8.reshape((1, 160, 3))

# op _00VP reshape_eval
# LANG: eel_coll_pts_coords --> _00VQ
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2013__00VQ = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00VV reshape_eval
# LANG: _00V9 --> _00VW
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2016__00VW = v1991__00V9.reshape((1, 160, 3))

# op _00Vb reshape_eval
# LANG: eel_coll_pts_coords --> _00Vc
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1993__00Vc = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00Vh reshape_eval
# LANG: _00V7 --> _00Vi
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1996__00Vi = v1989__00V7.reshape((1, 160, 3))

# op _00Vv reshape_eval
# LANG: eel_coll_pts_coords --> _00Vw
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2003__00Vw = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00PS_power_combination_eval
# LANG: _00PR --> _00PT
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1821__00PT = (v1820__00PR**2)
v1821__00PT = v1821__00PT.reshape((1, 44160, 3))

# op _00QN_linear_combination_eval
# LANG: _00QG, _00QM --> _00QO
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1850__00QO = v1846__00QG+-1*v1849__00QM

# op _00Qb_power_combination_eval
# LANG: _00Qa --> _00Qc
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1831__00Qc = (v1830__00Qa**2)
v1831__00Qc = v1831__00Qc.reshape((1, 44160, 3))

# op _00Qv_power_combination_eval
# LANG: _00Qu --> _00Qw
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1841__00Qw = (v1840__00Qu**2)
v1841__00Qw = v1841__00Qw.reshape((1, 44160, 3))

# op _00VD expand_array_eval
# LANG: _00VC --> _00VE
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2007__00VE = np.einsum('acd,b->abcd', v2006__00VC.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00VR expand_array_eval
# LANG: _00VQ --> _00VS
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2014__00VS = np.einsum('abd,c->abcd', v2013__00VQ.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00VX expand_array_eval
# LANG: _00VW --> _00VY
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2017__00VY = np.einsum('acd,b->abcd', v2016__00VW.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00Vd expand_array_eval
# LANG: _00Vc --> _00Ve
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1994__00Ve = np.einsum('abd,c->abcd', v1993__00Vc.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00Vj expand_array_eval
# LANG: _00Vi --> _00Vk
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1997__00Vk = np.einsum('acd,b->abcd', v1996__00Vi.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00Vx expand_array_eval
# LANG: _00Vw --> _00Vy
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2004__00Vy = np.einsum('abd,c->abcd', v2003__00Vw.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00W8 reshape_eval
# LANG: eel_coll_pts_coords --> _00W9
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2023__00W9 = v1987_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00We reshape_eval
# LANG: _00Va --> _00Wf
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2026__00Wf = v1992__00Va.reshape((1, 160, 3))

# op _00PU_single_tensor_sum_with_axis_eval
# LANG: _00PT --> _00PV
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1822__00PV = np.sum(v1821__00PT, axis = (2,)).reshape((1, 44160))

# op _00QP_power_combination_eval
# LANG: _00QO --> _00QQ
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1851__00QQ = (v1850__00QO**2)
v1851__00QQ = v1851__00QQ.reshape((1, 44160, 3))

# op _00Qd_single_tensor_sum_with_axis_eval
# LANG: _00Qc --> _00Qe
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1832__00Qe = np.sum(v1831__00Qc, axis = (2,)).reshape((1, 44160))

# op _00Qx_single_tensor_sum_with_axis_eval
# LANG: _00Qw --> _00Qy
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1842__00Qy = np.sum(v1841__00Qw, axis = (2,)).reshape((1, 44160))

# op _00VF reshape_eval
# LANG: _00VE --> _00VG
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2008__00VG = v2007__00VE.reshape((1, 25600, 3))

# op _00VT reshape_eval
# LANG: _00VS --> _00VU
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2015__00VU = v2014__00VS.reshape((1, 25600, 3))

# op _00VZ reshape_eval
# LANG: _00VY --> _00V_
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2018__00V_ = v2017__00VY.reshape((1, 25600, 3))

# op _00Vf reshape_eval
# LANG: _00Ve --> _00Vg
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1995__00Vg = v1994__00Ve.reshape((1, 25600, 3))

# op _00Vl reshape_eval
# LANG: _00Vk --> _00Vm
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1998__00Vm = v1997__00Vk.reshape((1, 25600, 3))

# op _00Vz reshape_eval
# LANG: _00Vy --> _00VA
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2005__00VA = v2004__00Vy.reshape((1, 25600, 3))

# op _00Wa expand_array_eval
# LANG: _00W9 --> _00Wb
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2024__00Wb = np.einsum('abd,c->abcd', v2023__00W9.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00Wg expand_array_eval
# LANG: _00Wf --> _00Wh
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2027__00Wh = np.einsum('acd,b->abcd', v2026__00Wf.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00PW_power_combination_eval
# LANG: _00PV --> _00PX
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1823__00PX = (v1822__00PV**0.5)
v1823__00PX = v1823__00PX.reshape((1, 44160))

# op _00QR_single_tensor_sum_with_axis_eval
# LANG: _00QQ --> _00QS
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1852__00QS = np.sum(v1851__00QQ, axis = (2,)).reshape((1, 44160))

# op _00Qf_power_combination_eval
# LANG: _00Qe --> _00Qg
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1833__00Qg = (v1832__00Qe**0.5)
v1833__00Qg = v1833__00Qg.reshape((1, 44160))

# op _00Qz_power_combination_eval
# LANG: _00Qy --> _00QA
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1843__00QA = (v1842__00Qy**0.5)
v1843__00QA = v1843__00QA.reshape((1, 44160))

# op _00VH_linear_combination_eval
# LANG: _00VA, _00VG --> _00VI
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2009__00VI = v2005__00VA+-1*v2008__00VG

# op _00Vn_linear_combination_eval
# LANG: _00Vg, _00Vm --> _00Vo
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v1999__00Vo = v1995__00Vg+-1*v1998__00Vm

# op _00W0_linear_combination_eval
# LANG: _00VU, _00V_ --> _00W1
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2019__00W1 = v2015__00VU+-1*v2018__00V_

# op _00Wc reshape_eval
# LANG: _00Wb --> _00Wd
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2025__00Wd = v2024__00Wb.reshape((1, 25600, 3))

# op _00Wi reshape_eval
# LANG: _00Wh --> _00Wj
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2028__00Wj = v2027__00Wh.reshape((1, 25600, 3))

# op _00QT_power_combination_eval
# LANG: _00QS --> _00QU
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1853__00QU = (v1852__00QS**0.5)
v1853__00QU = v1853__00QU.reshape((1, 44160))

# op _00QZ_power_combination_eval
# LANG: _00PR, _00Qa --> _00Q_
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1856__00Q_ = (v1820__00PR)*(v1830__00Qa)
v1856__00Q_ = v1856__00Q_.reshape((1, 44160, 3))

# op _00R2_power_combination_eval
# LANG: _00PX --> _00R3
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1858__00R3 = (v1823__00PX**2)
v1858__00R3 = v1858__00R3.reshape((1, 44160))

# op _00R4_power_combination_eval
# LANG: _00Qg --> _00R5
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1859__00R5 = (v1833__00Qg**2)
v1859__00R5 = v1859__00R5.reshape((1, 44160))

# op _00RA_power_combination_eval
# LANG: _00PX --> _00RB
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1875__00RB = (v1823__00PX)
v1875__00RB = (v1875__00RB*_00RA_coeff).reshape((1, 44160))

# op _00RW_power_combination_eval
# LANG: _00Qu, _00Qa --> _00RX
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1886__00RX = (v1830__00Qa)*(v1840__00Qu)
v1886__00RX = v1886__00RX.reshape((1, 44160, 3))

# op _00R__power_combination_eval
# LANG: _00Qg --> _00S0
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1888__00S0 = (v1833__00Qg**2)
v1888__00S0 = v1888__00S0.reshape((1, 44160))

# op _00S1_power_combination_eval
# LANG: _00QA --> _00S2
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1889__00S2 = (v1843__00QA**2)
v1889__00S2 = v1889__00S2.reshape((1, 44160))

# op _00Sx_power_combination_eval
# LANG: _00Qg --> _00Sy
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1905__00Sy = (v1833__00Qg)
v1905__00Sy = (v1905__00Sy*_00Sx_coeff).reshape((1, 44160))

# op _00VJ_power_combination_eval
# LANG: _00VI --> _00VK
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2010__00VK = (v2009__00VI**2)
v2010__00VK = v2010__00VK.reshape((1, 25600, 3))

# op _00Vp_power_combination_eval
# LANG: _00Vo --> _00Vq
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2000__00Vq = (v1999__00Vo**2)
v2000__00Vq = v2000__00Vq.reshape((1, 25600, 3))

# op _00W2_power_combination_eval
# LANG: _00W1 --> _00W3
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2020__00W3 = (v2019__00W1**2)
v2020__00W3 = v2020__00W3.reshape((1, 25600, 3))

# op _00Wk_linear_combination_eval
# LANG: _00Wd, _00Wj --> _00Wl
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2029__00Wl = v2025__00Wd+-1*v2028__00Wj

# op _00R0_single_tensor_sum_with_axis_eval
# LANG: _00Q_ --> _00R1
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1857__00R1 = np.sum(v1856__00Q_, axis = (2,)).reshape((1, 44160))

# op _00R8_linear_combination_eval
# LANG: _00R3 --> _00R9
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1861__00R9 = _00R8_constant+v1858__00R3

# op _00RC_power_combination_eval
# LANG: _00Qg, _00RB --> _00RD
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1876__00RD = (v1875__00RB)*(v1833__00Qg)
v1876__00RD = v1876__00RD.reshape((1, 44160))

# op _00RY_single_tensor_sum_with_axis_eval
# LANG: _00RX --> _00RZ
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1887__00RZ = np.sum(v1886__00RX, axis = (2,)).reshape((1, 44160))

# op _00Ri_linear_combination_eval
# LANG: _00R5 --> _00Rj
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1866__00Rj = _00Ri_constant+v1859__00R5

# op _00Ry_linear_combination_eval
# LANG: _00R3, _00R5 --> _00Rz
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1874__00Rz = v1858__00R3+v1859__00R5

# op _00S5_linear_combination_eval
# LANG: _00S0 --> _00S6
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1891__00S6 = _00S5_constant+v1888__00S0

# op _00ST_power_combination_eval
# LANG: _00QO, _00Qu --> _00SU
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1916__00SU = (v1840__00Qu)*(v1850__00QO)
v1916__00SU = v1916__00SU.reshape((1, 44160, 3))

# op _00SX_power_combination_eval
# LANG: _00QA --> _00SY
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1918__00SY = (v1843__00QA**2)
v1918__00SY = v1918__00SY.reshape((1, 44160))

# op _00SZ_power_combination_eval
# LANG: _00QU --> _00S_
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1919__00S_ = (v1853__00QU**2)
v1919__00S_ = v1919__00S_.reshape((1, 44160))

# op _00Sf_linear_combination_eval
# LANG: _00S2 --> _00Sg
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1896__00Sg = _00Sf_constant+v1889__00S2

# op _00Sv_linear_combination_eval
# LANG: _00S0, _00S2 --> _00Sw
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1904__00Sw = v1888__00S0+v1889__00S2

# op _00Sz_power_combination_eval
# LANG: _00QA, _00Sy --> _00SA
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1906__00SA = (v1905__00Sy)*(v1843__00QA)
v1906__00SA = v1906__00SA.reshape((1, 44160))

# op _00Tu_power_combination_eval
# LANG: _00QA --> _00Tv
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1935__00Tv = (v1843__00QA)
v1935__00Tv = (v1935__00Tv*_00Tu_coeff).reshape((1, 44160))

# op _00VL_single_tensor_sum_with_axis_eval
# LANG: _00VK --> _00VM
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2011__00VM = np.sum(v2010__00VK, axis = (2,)).reshape((1, 25600))

# op _00Vr_single_tensor_sum_with_axis_eval
# LANG: _00Vq --> _00Vs
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2001__00Vs = np.sum(v2000__00Vq, axis = (2,)).reshape((1, 25600))

# op _00W4_single_tensor_sum_with_axis_eval
# LANG: _00W3 --> _00W5
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2021__00W5 = np.sum(v2020__00W3, axis = (2,)).reshape((1, 25600))

# op _00Wm_power_combination_eval
# LANG: _00Wl --> _00Wn
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2030__00Wn = (v2029__00Wl**2)
v2030__00Wn = v2030__00Wn.reshape((1, 25600, 3))

# op _00RE_linear_combination_eval
# LANG: _00Rz, _00RD --> _00RF
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1877__00RF = v1874__00Rz+-1*v1876__00RD

# op _00Ra_linear_combination_eval
# LANG: _00R9 --> _00Rb
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1862__00Rb = _00Ra_constant+v1861__00R9

# op _00Rk_linear_combination_eval
# LANG: _00Rj --> _00Rl
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1867__00Rl = _00Rk_constant+v1866__00Rj

# op _00Rs_power_combination_eval
# LANG: _00R3, _00R5 --> _00Rt
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1871__00Rt = (v1858__00R3)*(v1859__00R5)
v1871__00Rt = v1871__00Rt.reshape((1, 44160))

# op _00Ru_power_combination_eval
# LANG: _00R1 --> _00Rv
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1872__00Rv = (v1857__00R1**2)
v1872__00Rv = v1872__00Rv.reshape((1, 44160))

# op _00S7_linear_combination_eval
# LANG: _00S6 --> _00S8
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1892__00S8 = _00S7_constant+v1891__00S6

# op _00SB_linear_combination_eval
# LANG: _00Sw, _00SA --> _00SC
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1907__00SC = v1904__00Sw+-1*v1906__00SA

# op _00SV_single_tensor_sum_with_axis_eval
# LANG: _00SU --> _00SW
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1917__00SW = np.sum(v1916__00SU, axis = (2,)).reshape((1, 44160))

# op _00Sh_linear_combination_eval
# LANG: _00Sg --> _00Si
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1897__00Si = _00Sh_constant+v1896__00Sg

# op _00Sp_power_combination_eval
# LANG: _00S0, _00S2 --> _00Sq
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1901__00Sq = (v1888__00S0)*(v1889__00S2)
v1901__00Sq = v1901__00Sq.reshape((1, 44160))

# op _00Sr_power_combination_eval
# LANG: _00RZ --> _00Ss
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1902__00Ss = (v1887__00RZ**2)
v1902__00Ss = v1902__00Ss.reshape((1, 44160))

# op _00T2_linear_combination_eval
# LANG: _00SY --> _00T3
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1921__00T3 = _00T2_constant+v1918__00SY

# op _00TQ_power_combination_eval
# LANG: _00QO, _00PR --> _00TR
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1946__00TR = (v1850__00QO)*(v1820__00PR)
v1946__00TR = v1946__00TR.reshape((1, 44160, 3))

# op _00TU_power_combination_eval
# LANG: _00QU --> _00TV
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1948__00TV = (v1853__00QU**2)
v1948__00TV = v1948__00TV.reshape((1, 44160))

# op _00TW_power_combination_eval
# LANG: _00PX --> _00TX
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1949__00TX = (v1823__00PX**2)
v1949__00TX = v1949__00TX.reshape((1, 44160))

# op _00Tc_linear_combination_eval
# LANG: _00S_ --> _00Td
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1926__00Td = _00Tc_constant+v1919__00S_

# op _00Ts_linear_combination_eval
# LANG: _00SY, _00S_ --> _00Tt
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1934__00Tt = v1918__00SY+v1919__00S_

# op _00Tw_power_combination_eval
# LANG: _00QU, _00Tv --> _00Tx
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1936__00Tx = (v1935__00Tv)*(v1853__00QU)
v1936__00Tx = v1936__00Tx.reshape((1, 44160))

# op _00Ur_power_combination_eval
# LANG: _00QU --> _00Us
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1965__00Us = (v1853__00QU)
v1965__00Us = (v1965__00Us*_00Ur_coeff).reshape((1, 44160))

# op _00VN_power_combination_eval
# LANG: _00VM --> _00VO
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2012__00VO = (v2011__00VM**0.5)
v2012__00VO = v2012__00VO.reshape((1, 25600))

# op _00Vt_power_combination_eval
# LANG: _00Vs --> _00Vu
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2002__00Vu = (v2001__00Vs**0.5)
v2002__00Vu = v2002__00Vu.reshape((1, 25600))

# op _00W6_power_combination_eval
# LANG: _00W5 --> _00W7
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2022__00W7 = (v2021__00W5**0.5)
v2022__00W7 = v2022__00W7.reshape((1, 25600))

# op _00WW_power_combination_eval
# LANG: _00W1, _00VI --> _00WX
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2048__00WX = (v2009__00VI)*(v2019__00W1)
v2048__00WX = v2048__00WX.reshape((1, 25600, 3))

# op _00Wo_single_tensor_sum_with_axis_eval
# LANG: _00Wn --> _00Wp
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2031__00Wp = np.sum(v2030__00Wn, axis = (2,)).reshape((1, 25600))

# op _00Ww_power_combination_eval
# LANG: _00Vo, _00VI --> _00Wx
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2035__00Wx = (v1999__00Vo)*(v2009__00VI)
v2035__00Wx = v2035__00Wx.reshape((1, 25600, 3))

# op _00R6_linear_combination_eval
# LANG: _00R3, _00R1 --> _00R7
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1860__00R7 = v1858__00R3+-1*v1857__00R1

# op _00RG_power_combination_eval
# LANG: _00RF --> _00RH
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1878__00RH = (v1877__00RF)
v1878__00RH = (v1878__00RH*_00RG_coeff).reshape((1, 44160))

# op _00Rc_power_combination_eval
# LANG: _00Rb --> _00Rd
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1863__00Rd = (v1862__00Rb**0.5)
v1863__00Rd = v1863__00Rd.reshape((1, 44160))

# op _00Rg_linear_combination_eval
# LANG: _00R5, _00R1 --> _00Rh
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1865__00Rh = v1859__00R5+-1*v1857__00R1

# op _00Rm_power_combination_eval
# LANG: _00Rl --> _00Rn
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1868__00Rn = (v1867__00Rl**0.5)
v1868__00Rn = v1868__00Rn.reshape((1, 44160))

# op _00Rw_linear_combination_eval
# LANG: _00Rt, _00Rv --> _00Rx
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1873__00Rx = v1871__00Rt+-1*v1872__00Rv

# op _00S3_linear_combination_eval
# LANG: _00S0, _00RZ --> _00S4
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1890__00S4 = v1888__00S0+-1*v1887__00RZ

# op _00S9_power_combination_eval
# LANG: _00S8 --> _00Sa
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1893__00Sa = (v1892__00S8**0.5)
v1893__00Sa = v1893__00Sa.reshape((1, 44160))

# op _00SD_power_combination_eval
# LANG: _00SC --> _00SE
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1908__00SE = (v1907__00SC)
v1908__00SE = (v1908__00SE*_00SD_coeff).reshape((1, 44160))

# op _00Sd_linear_combination_eval
# LANG: _00S2, _00RZ --> _00Se
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1895__00Se = v1889__00S2+-1*v1887__00RZ

# op _00Sj_power_combination_eval
# LANG: _00Si --> _00Sk
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1898__00Sk = (v1897__00Si**0.5)
v1898__00Sk = v1898__00Sk.reshape((1, 44160))

# op _00St_linear_combination_eval
# LANG: _00Sq, _00Ss --> _00Su
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1903__00Su = v1901__00Sq+-1*v1902__00Ss

# op _00T4_linear_combination_eval
# LANG: _00T3 --> _00T5
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1922__00T5 = _00T4_constant+v1921__00T3

# op _00TS_single_tensor_sum_with_axis_eval
# LANG: _00TR --> _00TT
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1947__00TT = np.sum(v1946__00TR, axis = (2,)).reshape((1, 44160))

# op _00T__linear_combination_eval
# LANG: _00TV --> _00U0
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1951__00U0 = _00T__constant+v1948__00TV

# op _00Te_linear_combination_eval
# LANG: _00Td --> _00Tf
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1927__00Tf = _00Te_constant+v1926__00Td

# op _00Tm_power_combination_eval
# LANG: _00SY, _00S_ --> _00Tn
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1931__00Tn = (v1918__00SY)*(v1919__00S_)
v1931__00Tn = v1931__00Tn.reshape((1, 44160))

# op _00To_power_combination_eval
# LANG: _00SW --> _00Tp
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1932__00Tp = (v1917__00SW**2)
v1932__00Tp = v1932__00Tp.reshape((1, 44160))

# op _00Ty_linear_combination_eval
# LANG: _00Tt, _00Tx --> _00Tz
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1937__00Tz = v1934__00Tt+-1*v1936__00Tx

# op _00U9_linear_combination_eval
# LANG: _00TX --> _00Ua
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1956__00Ua = _00U9_constant+v1949__00TX

# op _00Up_linear_combination_eval
# LANG: _00TV, _00TX --> _00Uq
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1964__00Uq = v1948__00TV+v1949__00TX

# op _00Ut_power_combination_eval
# LANG: _00Us, _00PX --> _00Uu
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1966__00Uu = (v1965__00Us)*(v1823__00PX)
v1966__00Uu = v1966__00Uu.reshape((1, 44160))

# op _00WA_power_combination_eval
# LANG: _00Vu, _00VO --> _00WB
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2037__00WB = (v2002__00Vu)*(v2012__00VO)
v2037__00WB = v2037__00WB.reshape((1, 25600))

# op _00WY_single_tensor_sum_with_axis_eval
# LANG: _00WX --> _00WZ
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2049__00WZ = np.sum(v2048__00WX, axis = (2,)).reshape((1, 25600))

# op _00W__power_combination_eval
# LANG: _00W7, _00VO --> _00X0
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2050__00X0 = (v2012__00VO)*(v2022__00W7)
v2050__00X0 = v2050__00X0.reshape((1, 25600))

# op _00Wq_power_combination_eval
# LANG: _00Wp --> _00Wr
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2032__00Wr = (v2031__00Wp**0.5)
v2032__00Wr = v2032__00Wr.reshape((1, 25600))

# op _00Wy_single_tensor_sum_with_axis_eval
# LANG: _00Wx --> _00Wz
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2036__00Wz = np.sum(v2035__00Wx, axis = (2,)).reshape((1, 25600))

# op _00Xl_power_combination_eval
# LANG: _00Wl, _00W1 --> _00Xm
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2061__00Xm = (v2019__00W1)*(v2029__00Wl)
v2061__00Xm = v2061__00Xm.reshape((1, 25600, 3))

# op _00OX_decompose_eval
# LANG: eel --> _00P2, _00OY, _00OZ, _00P1
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1786__00OY = ((v1785_eel.flatten())[src_indices__00OY__00OX]).reshape((1, 40, 4, 3))
v1787__00OZ = ((v1785_eel.flatten())[src_indices__00OZ__00OX]).reshape((1, 40, 4, 3))
v1789__00P1 = ((v1785_eel.flatten())[src_indices__00P1__00OX]).reshape((1, 40, 4, 3))
v1790__00P2 = ((v1785_eel.flatten())[src_indices__00P2__00OX]).reshape((1, 40, 4, 3))

# op _00RI_linear_combination_eval
# LANG: _00Rx, _00RH --> _00RJ
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1879__00RJ = v1873__00Rx+v1878__00RH

# op _00Re_power_combination_eval
# LANG: _00R7, _00Rd --> _00Rf
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1864__00Rf = (v1860__00R7)*(v1863__00Rd**-1)
v1864__00Rf = v1864__00Rf.reshape((1, 44160))

# op _00Ro_power_combination_eval
# LANG: _00Rh, _00Rn --> _00Rp
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1869__00Rp = (v1865__00Rh)*(v1868__00Rn**-1)
v1869__00Rp = v1869__00Rp.reshape((1, 44160))

# op _00SF_linear_combination_eval
# LANG: _00Su, _00SE --> _00SG
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1909__00SG = v1903__00Su+v1908__00SE

# op _00Sb_power_combination_eval
# LANG: _00S4, _00Sa --> _00Sc
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1894__00Sc = (v1890__00S4)*(v1893__00Sa**-1)
v1894__00Sc = v1894__00Sc.reshape((1, 44160))

# op _00Sl_power_combination_eval
# LANG: _00Se, _00Sk --> _00Sm
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1899__00Sm = (v1895__00Se)*(v1898__00Sk**-1)
v1899__00Sm = v1899__00Sm.reshape((1, 44160))

# op _00T0_linear_combination_eval
# LANG: _00SY, _00SW --> _00T1
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1920__00T1 = v1918__00SY+-1*v1917__00SW

# op _00T6_power_combination_eval
# LANG: _00T5 --> _00T7
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1923__00T7 = (v1922__00T5**0.5)
v1923__00T7 = v1923__00T7.reshape((1, 44160))

# op _00TA_power_combination_eval
# LANG: _00Tz --> _00TB
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1938__00TB = (v1937__00Tz)
v1938__00TB = (v1938__00TB*_00TA_coeff).reshape((1, 44160))

# op _00Ta_linear_combination_eval
# LANG: _00S_, _00SW --> _00Tb
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1925__00Tb = v1919__00S_+-1*v1917__00SW

# op _00Tg_power_combination_eval
# LANG: _00Tf --> _00Th
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1928__00Th = (v1927__00Tf**0.5)
v1928__00Th = v1928__00Th.reshape((1, 44160))

# op _00Tq_linear_combination_eval
# LANG: _00Tn, _00Tp --> _00Tr
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1933__00Tr = v1931__00Tn+-1*v1932__00Tp

# op _00U1_linear_combination_eval
# LANG: _00U0 --> _00U2
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1952__00U2 = _00U1_constant+v1951__00U0

# op _00Ub_linear_combination_eval
# LANG: _00Ua --> _00Uc
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1957__00Uc = _00Ub_constant+v1956__00Ua

# op _00Uj_power_combination_eval
# LANG: _00TV, _00TX --> _00Uk
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1961__00Uk = (v1948__00TV)*(v1949__00TX)
v1961__00Uk = v1961__00Uk.reshape((1, 44160))

# op _00Ul_power_combination_eval
# LANG: _00TT --> _00Um
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1962__00Um = (v1947__00TT**2)
v1962__00Um = v1962__00Um.reshape((1, 44160))

# op _00Uv_linear_combination_eval
# LANG: _00Uq, _00Uu --> _00Uw
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1967__00Uw = v1964__00Uq+-1*v1966__00Uu

# op _00WC_linear_combination_eval
# LANG: _00WB, _00Wz --> _00WD
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2038__00WD = v2037__00WB+v2036__00Wz

# op _00WG_power_combination_eval
# LANG: _00Vu --> _00WH
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2040__00WH = (v2002__00Vu**-1)
v2040__00WH = v2040__00WH.reshape((1, 25600))

# op _00WI_power_combination_eval
# LANG: _00VO --> _00WJ
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2041__00WJ = (v2012__00VO**-1)
v2041__00WJ = v2041__00WJ.reshape((1, 25600))

# op _00X1_linear_combination_eval
# LANG: _00X0, _00WZ --> _00X2
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2051__00X2 = v2050__00X0+v2049__00WZ

# op _00X5_power_combination_eval
# LANG: _00VO --> _00X6
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2053__00X6 = (v2012__00VO**-1)
v2053__00X6 = v2053__00X6.reshape((1, 25600))

# op _00X7_power_combination_eval
# LANG: _00W7 --> _00X8
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2054__00X8 = (v2022__00W7**-1)
v2054__00X8 = v2054__00X8.reshape((1, 25600))

# op _00XL_power_combination_eval
# LANG: _00Wl, _00Vo --> _00XM
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2074__00XM = (v2029__00Wl)*(v1999__00Vo)
v2074__00XM = v2074__00XM.reshape((1, 25600, 3))

# op _00Xn_single_tensor_sum_with_axis_eval
# LANG: _00Xm --> _00Xo
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2062__00Xo = np.sum(v2061__00Xm, axis = (2,)).reshape((1, 25600))

# op _00Xp_power_combination_eval
# LANG: _00Wr, _00W7 --> _00Xq
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2063__00Xq = (v2022__00W7)*(v2032__00Wr)
v2063__00Xq = v2063__00Xq.reshape((1, 25600))

# op _00O__linear_combination_eval
# LANG: _00OY, _00OZ --> _00P0
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1788__00P0 = v1786__00OY+-1*v1787__00OZ

# op _00P3_linear_combination_eval
# LANG: _00P1, _00P2 --> _00P4
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1791__00P4 = v1789__00P1+-1*v1790__00P2

# op _00RK_linear_combination_eval
# LANG: _00RJ --> _00RL
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1880__00RL = _00RK_constant+v1879__00RJ

# op _00Rq_linear_combination_eval
# LANG: _00Rf, _00Rp --> _00Rr
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1870__00Rr = v1864__00Rf+v1869__00Rp

# op _00SH_linear_combination_eval
# LANG: _00SG --> _00SI
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1910__00SI = _00SH_constant+v1909__00SG

# op _00Sn_linear_combination_eval
# LANG: _00Sc, _00Sm --> _00So
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1900__00So = v1894__00Sc+v1899__00Sm

# op _00T8_power_combination_eval
# LANG: _00T1, _00T7 --> _00T9
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1924__00T9 = (v1920__00T1)*(v1923__00T7**-1)
v1924__00T9 = v1924__00T9.reshape((1, 44160))

# op _00TC_linear_combination_eval
# LANG: _00Tr, _00TB --> _00TD
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1939__00TD = v1933__00Tr+v1938__00TB

# op _00TY_linear_combination_eval
# LANG: _00TV, _00TT --> _00TZ
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1950__00TZ = v1948__00TV+-1*v1947__00TT

# op _00Ti_power_combination_eval
# LANG: _00Tb, _00Th --> _00Tj
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1929__00Tj = (v1925__00Tb)*(v1928__00Th**-1)
v1929__00Tj = v1929__00Tj.reshape((1, 44160))

# op _00U3_power_combination_eval
# LANG: _00U2 --> _00U4
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1953__00U4 = (v1952__00U2**0.5)
v1953__00U4 = v1953__00U4.reshape((1, 44160))

# op _00U7_linear_combination_eval
# LANG: _00TX, _00TT --> _00U8
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1955__00U8 = v1949__00TX+-1*v1947__00TT

# op _00Ud_power_combination_eval
# LANG: _00Uc --> _00Ue
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1958__00Ue = (v1957__00Uc**0.5)
v1958__00Ue = v1958__00Ue.reshape((1, 44160))

# op _00Un_linear_combination_eval
# LANG: _00Uk, _00Um --> _00Uo
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1963__00Uo = v1961__00Uk+-1*v1962__00Um

# op _00Ux_power_combination_eval
# LANG: _00Uw --> _00Uy
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1968__00Uy = (v1967__00Uw)
v1968__00Uy = (v1968__00Uy*_00Ux_coeff).reshape((1, 44160))

# op _00WE_power_combination_eval
# LANG: _00WD --> _00WF
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2039__00WF = (v2038__00WD**-1)
v2039__00WF = v2039__00WF.reshape((1, 25600))

# op _00WK_linear_combination_eval
# LANG: _00WH, _00WJ --> _00WL
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2042__00WL = v2040__00WH+v2041__00WJ

# op _00X3_power_combination_eval
# LANG: _00X2 --> _00X4
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2052__00X4 = (v2051__00X2**-1)
v2052__00X4 = v2052__00X4.reshape((1, 25600))

# op _00X9_linear_combination_eval
# LANG: _00X6, _00X8 --> _00Xa
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2055__00Xa = v2053__00X6+v2054__00X8

# op _00XN_single_tensor_sum_with_axis_eval
# LANG: _00XM --> _00XO
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2075__00XO = np.sum(v2074__00XM, axis = (2,)).reshape((1, 25600))

# op _00XP_power_combination_eval
# LANG: _00Vu, _00Wr --> _00XQ
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2076__00XQ = (v2032__00Wr)*(v2002__00Vu)
v2076__00XQ = v2076__00XQ.reshape((1, 25600))

# op _00Xr_linear_combination_eval
# LANG: _00Xq, _00Xo --> _00Xs
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2064__00Xs = v2063__00Xq+v2062__00Xo

# op _00Xv_power_combination_eval
# LANG: _00W7 --> _00Xw
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2066__00Xw = (v2022__00W7**-1)
v2066__00Xw = v2066__00Xw.reshape((1, 25600))

# op _00Xx_power_combination_eval
# LANG: _00Wr --> _00Xy
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2067__00Xy = (v2032__00Wr**-1)
v2067__00Xy = v2067__00Xy.reshape((1, 25600))

# op _00P5 cross_product_eval
# LANG: _00P0, _00P4 --> _00P6
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1792__00P6 = np.cross(v1788__00P0, v1791__00P4, axisa = 3, axisb = 3, axisc = 3)

# op _00QV cross_product_eval
# LANG: _00PR, _00Qa --> _00QW
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1854__00QW = np.cross(v1820__00PR, v1830__00Qa, axisa = 2, axisb = 2, axisc = 2)

# op _00RM_power_combination_eval
# LANG: _00Rr, _00RL --> _00RN
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1881__00RN = (v1870__00Rr)*(v1880__00RL**-1)
v1881__00RN = v1881__00RN.reshape((1, 44160))

# op _00RS cross_product_eval
# LANG: _00Qu, _00Qa --> _00RT
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1884__00RT = np.cross(v1830__00Qa, v1840__00Qu, axisa = 2, axisb = 2, axisc = 2)

# op _00SJ_power_combination_eval
# LANG: _00So, _00SI --> _00SK
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1911__00SK = (v1900__00So)*(v1910__00SI**-1)
v1911__00SK = v1911__00SK.reshape((1, 44160))

# op _00TE_linear_combination_eval
# LANG: _00TD --> _00TF
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1940__00TF = _00TE_constant+v1939__00TD

# op _00Tk_linear_combination_eval
# LANG: _00T9, _00Tj --> _00Tl
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1930__00Tl = v1924__00T9+v1929__00Tj

# op _00U5_power_combination_eval
# LANG: _00TZ, _00U4 --> _00U6
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1954__00U6 = (v1950__00TZ)*(v1953__00U4**-1)
v1954__00U6 = v1954__00U6.reshape((1, 44160))

# op _00Uf_power_combination_eval
# LANG: _00U8, _00Ue --> _00Ug
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1959__00Ug = (v1955__00U8)*(v1958__00Ue**-1)
v1959__00Ug = v1959__00Ug.reshape((1, 44160))

# op _00Uz_linear_combination_eval
# LANG: _00Uo, _00Uy --> _00UA
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1969__00UA = v1963__00Uo+v1968__00Uy

# op _00WM_power_combination_eval
# LANG: _00WF, _00WL --> _00WN
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2043__00WN = (v2039__00WF)*(v2042__00WL)
v2043__00WN = v2043__00WN.reshape((1, 25600))

# op _00WS cross_product_eval
# LANG: _00W1, _00VI --> _00WT
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2046__00WT = np.cross(v2009__00VI, v2019__00W1, axisa = 2, axisb = 2, axisc = 2)

# op _00Ws cross_product_eval
# LANG: _00Vo, _00VI --> _00Wt
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2033__00Wt = np.cross(v1999__00Vo, v2009__00VI, axisa = 2, axisb = 2, axisc = 2)

# op _00XR_linear_combination_eval
# LANG: _00XQ, _00XO --> _00XS
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2077__00XS = v2076__00XQ+v2075__00XO

# op _00XV_power_combination_eval
# LANG: _00Wr --> _00XW
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2079__00XW = (v2032__00Wr**-1)
v2079__00XW = v2079__00XW.reshape((1, 25600))

# op _00XX_power_combination_eval
# LANG: _00Vu --> _00XY
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2080__00XY = (v2002__00Vu**-1)
v2080__00XY = v2080__00XY.reshape((1, 25600))

# op _00Xb_power_combination_eval
# LANG: _00X4, _00Xa --> _00Xc
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2056__00Xc = (v2052__00X4)*(v2055__00Xa)
v2056__00Xc = v2056__00Xc.reshape((1, 25600))

# op _00Xt_power_combination_eval
# LANG: _00Xs --> _00Xu
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2065__00Xu = (v2064__00Xs**-1)
v2065__00Xu = v2065__00Xu.reshape((1, 25600))

# op _00Xz_linear_combination_eval
# LANG: _00Xw, _00Xy --> _00XA
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2068__00XA = v2066__00Xw+v2067__00Xy

# op _00OB expand_array_eval
# LANG: eel_rot_ref --> _00OC
# SHAPES: (1, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1774__00OC = np.einsum('ad,bc->abcd', v1773_eel_rot_ref.reshape((1, 3)) ,np.ones((40, 4))).reshape((1, 40, 4, 3))

# op _00Oy_indexed_passthrough_eval
# LANG: p, q, r --> ang_vel
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1771_ang_vel__temp[i_v1768_p__00Oy_indexed_passthrough_eval] = v1768_p.flatten()
v1771_ang_vel = v1771_ang_vel__temp.copy()
v1771_ang_vel__temp[i_v1769_q__00Oy_indexed_passthrough_eval] = v1769_q.flatten()
v1771_ang_vel = v1771_ang_vel__temp.copy()
v1771_ang_vel__temp[i_v1770_r__00Oy_indexed_passthrough_eval] = v1770_r.flatten()
v1771_ang_vel = v1771_ang_vel__temp.copy()

# op _00P7_power_combination_eval
# LANG: _00P6 --> _00P8
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1793__00P8 = (v1792__00P6**2)
v1793__00P8 = v1793__00P8.reshape((1, 40, 4, 3))

# op _00QX_power_combination_eval
# LANG: _00QW --> _00QY
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1855__00QY = (v1854__00QW)
v1855__00QY = (v1855__00QY*_00QX_coeff).reshape((1, 44160, 3))

# op _00RO expand_array_eval
# LANG: _00RN --> _00RP
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1882__00RP = np.einsum('ab,c->abc', v1881__00RN.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00RU_power_combination_eval
# LANG: _00RT --> _00RV
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1885__00RV = (v1884__00RT)
v1885__00RV = (v1885__00RV*_00RU_coeff).reshape((1, 44160, 3))

# op _00SL expand_array_eval
# LANG: _00SK --> _00SM
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1912__00SM = np.einsum('ab,c->abc', v1911__00SK.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00SP cross_product_eval
# LANG: _00QO, _00Qu --> _00SQ
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1914__00SQ = np.cross(v1840__00Qu, v1850__00QO, axisa = 2, axisb = 2, axisc = 2)

# op _00TG_power_combination_eval
# LANG: _00Tl, _00TF --> _00TH
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1941__00TH = (v1930__00Tl)*(v1940__00TF**-1)
v1941__00TH = v1941__00TH.reshape((1, 44160))

# op _00UB_linear_combination_eval
# LANG: _00UA --> _00UC
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1970__00UC = _00UB_constant+v1969__00UA

# op _00Uh_linear_combination_eval
# LANG: _00U6, _00Ug --> _00Ui
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1960__00Ui = v1954__00U6+v1959__00Ug

# op _00WO expand_array_eval
# LANG: _00WN --> _00WP
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2044__00WP = np.einsum('ab,c->abc', v2043__00WN.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00WU_power_combination_eval
# LANG: _00WT --> _00WV
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2047__00WV = (v2046__00WT)
v2047__00WV = (v2047__00WV*_00WU_coeff).reshape((1, 25600, 3))

# op _00Wu_power_combination_eval
# LANG: _00Wt --> _00Wv
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2034__00Wv = (v2033__00Wt)
v2034__00Wv = (v2034__00Wv*_00Wu_coeff).reshape((1, 25600, 3))

# op _00XB_power_combination_eval
# LANG: _00Xu, _00XA --> _00XC
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2069__00XC = (v2065__00Xu)*(v2068__00XA)
v2069__00XC = v2069__00XC.reshape((1, 25600))

# op _00XT_power_combination_eval
# LANG: _00XS --> _00XU
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2078__00XU = (v2077__00XS**-1)
v2078__00XU = v2078__00XU.reshape((1, 25600))

# op _00XZ_linear_combination_eval
# LANG: _00XW, _00XY --> _00X_
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2081__00X_ = v2079__00XW+v2080__00XY

# op _00Xd expand_array_eval
# LANG: _00Xc --> _00Xe
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2057__00Xe = np.einsum('ab,c->abc', v2056__00Xc.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00Xh cross_product_eval
# LANG: _00Wl, _00W1 --> _00Xi
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2059__00Xi = np.cross(v2019__00W1, v2029__00Wl, axisa = 2, axisb = 2, axisc = 2)

# op _00OD_linear_combination_eval
# LANG: _00OC, eel_coll_pts_coords --> _00OE
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1775__00OE = v1987_eel_coll_pts_coords+-1*v1774__00OC

# op _00OF expand_array_eval
# LANG: ang_vel --> _00OG
# SHAPES: (1, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1776__00OG = np.einsum('ad,bc->abcd', v1771_ang_vel.reshape((1, 3)) ,np.ones((40, 4))).reshape((1, 40, 4, 3))

# op _00P9_single_tensor_sum_with_axis_eval
# LANG: _00P8 --> _00Pa
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1794__00Pa = np.sum(v1793__00P8, axis = (3,)).reshape((1, 40, 4))

# op _00RQ_power_combination_eval
# LANG: _00RP, _00QY --> _00RR
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1883__00RR = (v1882__00RP)*(v1855__00QY)
v1883__00RR = v1883__00RR.reshape((1, 44160, 3))

# op _00SN_power_combination_eval
# LANG: _00SM, _00RV --> _00SO
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1913__00SO = (v1912__00SM)*(v1885__00RV)
v1913__00SO = v1913__00SO.reshape((1, 44160, 3))

# op _00SR_power_combination_eval
# LANG: _00SQ --> _00SS
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1915__00SS = (v1914__00SQ)
v1915__00SS = (v1915__00SS*_00SR_coeff).reshape((1, 44160, 3))

# op _00TI expand_array_eval
# LANG: _00TH --> _00TJ
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1942__00TJ = np.einsum('ab,c->abc', v1941__00TH.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00TM cross_product_eval
# LANG: _00QO, _00PR --> _00TN
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1944__00TN = np.cross(v1850__00QO, v1820__00PR, axisa = 2, axisb = 2, axisc = 2)

# op _00UD_power_combination_eval
# LANG: _00Ui, _00UC --> _00UE
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1971__00UE = (v1960__00Ui)*(v1970__00UC**-1)
v1971__00UE = v1971__00UE.reshape((1, 44160))

# op _00WQ_power_combination_eval
# LANG: _00WP, _00Wv --> _00WR
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2045__00WR = (v2044__00WP)*(v2034__00Wv)
v2045__00WR = v2045__00WR.reshape((1, 25600, 3))

# op _00XD expand_array_eval
# LANG: _00XC --> _00XE
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2070__00XE = np.einsum('ab,c->abc', v2069__00XC.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00XH cross_product_eval
# LANG: _00Wl, _00Vo --> _00XI
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2072__00XI = np.cross(v2029__00Wl, v1999__00Vo, axisa = 2, axisb = 2, axisc = 2)

# op _00Xf_power_combination_eval
# LANG: _00Xe, _00WV --> _00Xg
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2058__00Xg = (v2057__00Xe)*(v2047__00WV)
v2058__00Xg = v2058__00Xg.reshape((1, 25600, 3))

# op _00Xj_power_combination_eval
# LANG: _00Xi --> _00Xk
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2060__00Xk = (v2059__00Xi)
v2060__00Xk = (v2060__00Xk*_00Xj_coeff).reshape((1, 25600, 3))

# op _00Y0_power_combination_eval
# LANG: _00XU, _00X_ --> _00Y1
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2082__00Y1 = (v2078__00XU)*(v2081__00X_)
v2082__00Y1 = v2082__00Y1.reshape((1, 25600))

# op _00OH cross_product_eval
# LANG: _00OG, _00OE --> _00OI
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1777__00OI = np.cross(v1776__00OG, v1775__00OE, axisa = 3, axisb = 3, axisc = 3)

# op _00Pb_power_combination_eval
# LANG: _00Pa --> _00Pc
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1795__00Pc = (v1794__00Pa**0.5)
v1795__00Pc = v1795__00Pc.reshape((1, 40, 4))

# op _00TK_power_combination_eval
# LANG: _00TJ, _00SS --> _00TL
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1943__00TL = (v1942__00TJ)*(v1915__00SS)
v1943__00TL = v1943__00TL.reshape((1, 44160, 3))

# op _00TO_power_combination_eval
# LANG: _00TN --> _00TP
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1945__00TP = (v1944__00TN)
v1945__00TP = (v1945__00TP*_00TO_coeff).reshape((1, 44160, 3))

# op _00UF expand_array_eval
# LANG: _00UE --> _00UG
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1972__00UG = np.einsum('ab,c->abc', v1971__00UE.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00UJ_linear_combination_eval
# LANG: _00RR, _00SO --> _00UK
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1974__00UK = v1883__00RR+v1913__00SO

# op _00XF_power_combination_eval
# LANG: _00XE, _00Xk --> _00XG
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2071__00XG = (v2070__00XE)*(v2060__00Xk)
v2071__00XG = v2071__00XG.reshape((1, 25600, 3))

# op _00XJ_power_combination_eval
# LANG: _00XI --> _00XK
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2073__00XK = (v2072__00XI)
v2073__00XK = (v2073__00XK*_00XJ_coeff).reshape((1, 25600, 3))

# op _00Y2 expand_array_eval
# LANG: _00Y1 --> _00Y3
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2083__00Y3 = np.einsum('ab,c->abc', v2082__00Y1.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00Y6_linear_combination_eval
# LANG: _00WR, _00Xg --> _00Y7
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2085__00Y7 = v2045__00WR+v2058__00Xg

# op _00OJ reshape_eval
# LANG: _00OI --> _00OK
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1778__00OK = v1777__00OI.reshape((1, 160, 3))

# op _00OL expand_array_eval
# LANG: frame_vel --> _00OM
# SHAPES: (1, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1779__00OM = np.einsum('ac,b->abc', v2097_frame_vel.reshape((1, 3)) ,np.ones((160,))).reshape((1, 160, 3))

# op _00Pd expand_array_eval
# LANG: _00Pc --> _00Pe
# SHAPES: (1, 40, 4) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v1796__00Pe = np.einsum('abc,d->abcd', v1795__00Pc.reshape((1, 40, 4)) ,np.ones((3,))).reshape((1, 40, 4, 3))

# op _00UH_power_combination_eval
# LANG: _00UG, _00TP --> _00UI
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1973__00UI = (v1972__00UG)*(v1945__00TP)
v1973__00UI = v1973__00UI.reshape((1, 44160, 3))

# op _00UL_linear_combination_eval
# LANG: _00UK, _00TL --> _00UM
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1975__00UM = v1974__00UK+v1943__00TL

# op _00Y4_power_combination_eval
# LANG: _00Y3, _00XK --> _00Y5
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2084__00Y5 = (v2083__00Y3)*(v2073__00XK)
v2084__00Y5 = v2084__00Y5.reshape((1, 25600, 3))

# op _00Y8_linear_combination_eval
# LANG: _00Y7, _00XG --> _00Y9
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2086__00Y9 = v2085__00Y7+v2071__00XG

# op _00OO_linear_combination_eval
# LANG: _00OK, _00OM --> _00OP
# SHAPES: (1, 160, 3), (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1781__00OP = v1778__00OK+v1779__00OM

# op _00OQ reshape_eval
# LANG: eel_coll_vel --> _00OR
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1782__00OR = v1780_eel_coll_vel.reshape((1, 160, 3))

# op _00Pf_power_combination_eval
# LANG: _00P6, _00Pe --> eel_bd_vtx_normals
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v2090_eel_bd_vtx_normals = (v1792__00P6)*(v1796__00Pe**-1)
v2090_eel_bd_vtx_normals = v2090_eel_bd_vtx_normals.reshape((1, 40, 4, 3))

# op _00UN_linear_combination_eval
# LANG: _00UM, _00UI --> aic_M00
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v1976_aic_M00 = v1975__00UM+v1973__00UI

# op _00Ya_linear_combination_eval
# LANG: _00Y9, _00Y5 --> aic_bd00
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v2087_aic_bd00 = v2086__00Y9+v2084__00Y5

# op _00OS_linear_combination_eval
# LANG: _00OP, _00OR --> _00OT
# SHAPES: (1, 160, 3), (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1783__00OT = v1781__00OP+v1782__00OR

# op _00Pu reshape_eval
# LANG: aic_M00 --> _00Pv
# SHAPES: (1, 44160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v1807__00Pv = v1976_aic_M00.reshape((1, 160, 276, 3))

# op _00US reshape_eval
# LANG: eel_bd_vtx_normals --> _00UT
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v1980__00UT = v2090_eel_bd_vtx_normals.reshape((1, 160, 3))

# op _00V1 reshape_eval
# LANG: aic_bd00 --> _00V2
# SHAPES: (1, 25600, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v1986__00V2 = v2087_aic_bd00.reshape((1, 160, 160, 3))

# op _00Yf reshape_eval
# LANG: eel_bd_vtx_normals --> _00Yg
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v2091__00Yg = v2090_eel_bd_vtx_normals.reshape((1, 160, 3))

# op _00OU_linear_combination_eval
# LANG: _00OT --> eel_kinematic_vel
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v1799_eel_kinematic_vel = -1*v1783__00OT

# op _00Pk reshape_eval
# LANG: eel_bd_vtx_normals --> _00Pl
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v1801__00Pl = v2090_eel_bd_vtx_normals.reshape((1, 160, 3))

# op _00Pw_indexed_passthrough_eval
# LANG: _00Pv --> aic_M
# SHAPES: (1, 160, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v1978_aic_M__temp[i_v1807__00Pv__00Pw_indexed_passthrough_eval] = v1807__00Pv.flatten()
v1978_aic_M = v1978_aic_M__temp.copy()

# op _00UU_indexed_passthrough_eval
# LANG: _00UT --> normal_concatenated_M_mat
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v1977_normal_concatenated_M_mat__temp[i_v1980__00UT__00UU_indexed_passthrough_eval] = v1980__00UT.flatten()
v1977_normal_concatenated_M_mat = v1977_normal_concatenated_M_mat__temp.copy()

# op _00V3_indexed_passthrough_eval
# LANG: _00V2 --> aic_bd
# SHAPES: (1, 160, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v2089_aic_bd__temp[i_v1986__00V2__00V3_indexed_passthrough_eval] = v1986__00V2.flatten()
v2089_aic_bd = v2089_aic_bd__temp.copy()

# op _00Yh_indexed_passthrough_eval
# LANG: _00Yg --> normal_concatenated_aic_bd_proj
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v2088_normal_concatenated_aic_bd_proj__temp[i_v2091__00Yg__00Yh_indexed_passthrough_eval] = v2091__00Yg.flatten()
v2088_normal_concatenated_aic_bd_proj = v2088_normal_concatenated_aic_bd_proj__temp.copy()

# op _00Pm_custom_explicit_eval
# LANG: _00Pl, eel_kinematic_vel --> b
# SHAPES: (1, 160, 3), (1, 160, 3) --> (1, 160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
temp = _00Pm_custom_explicit_func_b.solve(v1799_eel_kinematic_vel, v1801__00Pl)
v1802_b = temp[0].copy()

# op _00UV_custom_explicit_eval
# LANG: normal_concatenated_M_mat, aic_M --> M_mat
# SHAPES: (1, 160, 3), (1, 160, 276, 3) --> (1, 160, 276)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
temp = _00UV_custom_explicit_func_M_mat.solve(v1978_aic_M, v1977_normal_concatenated_M_mat)
v1981_M_mat = temp[0].copy()

# op _00Yi_custom_explicit_eval
# LANG: normal_concatenated_aic_bd_proj, aic_bd --> aic_bd_proj
# SHAPES: (1, 160, 3), (1, 160, 160, 3) --> (1, 160, 160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
temp = _00Yi_custom_explicit_func_aic_bd_proj.solve(v2089_aic_bd, v2088_normal_concatenated_aic_bd_proj)
v2092_aic_bd_proj = temp[0].copy()

# op _00NO_indexed_passthrough_eval
# LANG: eel_gamma_w --> gamma_w
# SHAPES: (1, 69, 4) --> (1, 69, 4)
# full namespace: combine_gamma_w
v1755_gamma_w__temp[i_v1743_eel_gamma_w__00NO_indexed_passthrough_eval] = v1743_eel_gamma_w.flatten()
v1755_gamma_w = v1755_gamma_w__temp.copy()

# op _00O9_newton_implict_eval
# LANG: aic_bd_proj, b, M_mat, gamma_w --> gamma_b
# SHAPES: (1, 160, 160), (1, 160), (1, 160, 276), (1, 69, 4) --> (1, 160)
# full namespace: solve_gamma_b_group
_00O9_newton.set_guess(initial_guess_v2093_gamma_b)
_00O9_newton_out = _00O9_newton.solve(v2092_aic_bd_proj, v1981_M_mat, v1755_gamma_w, v1802_b)
v2093_gamma_b = _00O9_newton_out[0]

# op _00Ys_linear_combination_eval
# LANG: frame_vel --> _00Yt
# SHAPES: (1, 3) --> (1, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v2098__00Yt = -1*v2097_frame_vel

# op _00Yu expand_array_eval
# LANG: _00Yt --> eel_wake_kinematic_vel
# SHAPES: (1, 3) --> (1, 69, 5, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v2099_eel_wake_kinematic_vel = np.einsum('ad,bc->abcd', v2098__00Yt.reshape((1, 3)) ,np.ones((69, 5))).reshape((1, 69, 5, 3))

# op _00Yp_linear_combination_eval
# LANG: eel_wake_kinematic_vel --> eel_wake_total_vel
# SHAPES: (1, 69, 5, 3) --> (1, 69, 5, 3)
# full namespace: ComputeWakeTotalVel
v2096_eel_wake_total_vel = v2099_eel_wake_kinematic_vel

# op _00JZ_decompose_eval
# LANG: eel_wake_total_vel --> _00KH, _00J_
# SHAPES: (1, 69, 5, 3) --> (1, 68, 5, 3), (1, 1, 5, 3)
# full namespace: 
v1605__00J_ = ((v2096_eel_wake_total_vel.flatten())[src_indices__00J___00JZ]).reshape((1, 1, 5, 3))
v1628__00KH = ((v2096_eel_wake_total_vel.flatten())[src_indices__00KH__00JZ]).reshape((1, 68, 5, 3))

# op _00Kp_power_combination_eval
# LANG: _00J_ --> _00Kq
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1618__00Kq = (v1605__00J_)
v1618__00Kq = (v1618__00Kq*_00Kp_coeff).reshape((1, 1, 5, 3))

# op _00JX_decompose_eval
# LANG: eel_bd_vtx_coords --> _00JY
# SHAPES: (1, 41, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1604__00JY = ((v1988_eel_bd_vtx_coords.flatten())[src_indices__00JY__00JX]).reshape((1, 1, 5, 3))

# op _00Kr_power_combination_eval
# LANG: _00Kq --> _00Ks
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1619__00Ks = (v1618__00Kq)
v1619__00Ks = (v1619__00Ks*_00Kr_coeff).reshape((1, 1, 5, 3))

# op _00K6_decompose_eval
# LANG: eel_wake_coords --> _00KE, _00K7, _00KB
# SHAPES: (1, 69, 5, 3) --> (1, 68, 5, 3), (1, 1, 5, 3), (1, 68, 5, 3)
# full namespace: 
v1609__00K7 = ((v1761_eel_wake_coords.flatten())[src_indices__00K7__00K6]).reshape((1, 1, 5, 3))
v1624__00KB = ((v1761_eel_wake_coords.flatten())[src_indices__00KB__00K6]).reshape((1, 68, 5, 3))
v1626__00KE = ((v1761_eel_wake_coords.flatten())[src_indices__00KE__00K6]).reshape((1, 68, 5, 3))

# op _00Kt_linear_combination_eval
# LANG: _00JY, _00Ks --> _00Ku
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1620__00Ku = v1604__00JY+v1619__00Ks

# op _00Mf_linear_combination_eval
# LANG: _00Lz, _00LC --> _00Mg
# SHAPES: (1, 40, 5, 3), (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v1688__00Mg = v1662__00Lz+-1*v1664__00LC

# op _00KY_power_combination_eval
# LANG: u --> _00KZ
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1642__00KZ = (v1632_u**2)
v1642__00KZ = v1642__00KZ.reshape((1, 1))

# op _00K__power_combination_eval
# LANG: v --> _00L0
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1643__00L0 = (v1633_v**2)
v1643__00L0 = v1643__00L0.reshape((1, 1))

# op _00Kv_linear_combination_eval
# LANG: _00Ku, _00K7 --> _00Kw
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1621__00Kw = v1620__00Ku+-1*v1609__00K7

# op _00Lv_linear_combination_eval
# LANG: eel --> _00Lw
# SHAPES: (1, 41, 5, 3) --> (1, 41, 5, 3)
# full namespace: MeshPreprocessing_comp
v1660__00Lw = v1785_eel

# op _00Mh pnorm_axis_eval
# LANG: _00Mg --> _00Mi
# SHAPES: (1, 40, 5, 3) --> (1, 40, 5)
# full namespace: MeshPreprocessing_comp
v1689__00Mi = np.sum(v1688__00Mg**2,axis=(3,))**(1 / 2)

# op _00Kx reshape_eval
# LANG: _00Kw --> _00Ky
# SHAPES: (1, 1, 5, 3) --> (1, 5, 3)
# full namespace: 
v1622__00Ky = v1621__00Kw.reshape((1, 5, 3))

# op _00L1_linear_combination_eval
# LANG: _00KZ, _00L0 --> _00L2
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1644__00L2 = v1642__00KZ+v1643__00L0

# op _00L3_power_combination_eval
# LANG: w --> _00L4
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1645__00L4 = (v1668_w**2)
v1645__00L4 = v1645__00L4.reshape((1, 1))

# op _00MD_decompose_eval
# LANG: _00Lw --> _00MJ, _00ME, _00MF, _00MI
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1702__00ME = ((v1660__00Lw.flatten())[src_indices__00ME__00MD]).reshape((1, 40, 4, 3))
v1703__00MF = ((v1660__00Lw.flatten())[src_indices__00MF__00MD]).reshape((1, 40, 4, 3))
v1705__00MI = ((v1660__00Lw.flatten())[src_indices__00MI__00MD]).reshape((1, 40, 4, 3))
v1706__00MJ = ((v1660__00Lw.flatten())[src_indices__00MJ__00MD]).reshape((1, 40, 4, 3))

# op _00MY_power_combination_eval
# LANG: _00MX --> _00MZ
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1715__00MZ = (v1714__00MX)
v1715__00MZ = (v1715__00MZ*_00MY_coeff).reshape((1, 40, 4, 3))

# op _00Mj_decompose_eval
# LANG: _00Mi --> _00Ml, _00Mk
# SHAPES: (1, 40, 5) --> (1, 40, 4), (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1690__00Mk = ((v1689__00Mi.flatten())[src_indices__00Mk__00Mj]).reshape((1, 40, 4))
v1691__00Ml = ((v1689__00Mi.flatten())[src_indices__00Ml__00Mj]).reshape((1, 40, 4))

# op _00N0_power_combination_eval
# LANG: _00M_ --> _00N1
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1717__00N1 = (v1716__00M_)
v1717__00N1 = (v1717__00N1*_00N0_coeff).reshape((1, 40, 4, 3))

# op _00K0_power_combination_eval
# LANG: _00J_ --> _00K1
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1606__00K1 = (v1605__00J_)
v1606__00K1 = (v1606__00K1*_00K0_coeff).reshape((1, 1, 5, 3))

# op _00Kz expand_array_eval
# LANG: _00Ky --> _00KA
# SHAPES: (1, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v1623__00KA = np.einsum('acd,b->abcd', v1622__00Ky.reshape((1, 5, 3)) ,np.ones((68,))).reshape((1, 68, 5, 3))

# op _00L5_linear_combination_eval
# LANG: _00L2, _00L4 --> v_inf_sq
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1734_v_inf_sq = v1644__00L2+v1645__00L4

# op _00MG_linear_combination_eval
# LANG: _00ME, _00MF --> _00MH
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1704__00MH = v1702__00ME+-1*v1703__00MF

# op _00MK_linear_combination_eval
# LANG: _00MI, _00MJ --> _00ML
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1707__00ML = v1705__00MI+-1*v1706__00MJ

# op _00Mm_linear_combination_eval
# LANG: _00Mk, _00Ml --> _00Mn
# SHAPES: (1, 40, 4), (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1692__00Mn = v1690__00Mk+v1691__00Ml

# op _00N2_linear_combination_eval
# LANG: _00MZ, _00N1 --> _00N3
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1718__00N3 = v1715__00MZ+v1717__00N1

# op _00N5_power_combination_eval
# LANG: _00N4 --> _00N6
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1720__00N6 = (v1719__00N4)
v1720__00N6 = (v1720__00N6*_00N5_coeff).reshape((1, 40, 4, 3))

# op _00Yl_decompose_eval
# LANG: gamma_b --> eel_gamma_b
# SHAPES: (1, 160) --> (1, 160)
# full namespace: seperate_gamma_b
v2094_eel_gamma_b = ((v2093_gamma_b.flatten())[src_indices_eel_gamma_b__00Yl]).reshape((1, 160))

# op _00JA_decompose_eval
# LANG: eel_gamma_b --> _00JB
# SHAPES: (1, 160) --> (1, 4)
# full namespace: 
v1590__00JB = ((v2094_eel_gamma_b.flatten())[src_indices__00JB__00JA]).reshape((1, 4))

# op _00K2_power_combination_eval
# LANG: _00K1 --> _00K3
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1607__00K3 = (v1606__00K1)
v1607__00K3 = (v1607__00K3*_00K2_coeff).reshape((1, 1, 5, 3))

# op _00KC_linear_combination_eval
# LANG: _00KA, _00KB --> _00KD
# SHAPES: (1, 68, 5, 3), (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v1625__00KD = v1623__00KA+v1624__00KB

# op _00MM cross_product_eval
# LANG: _00MH, _00ML --> _00MN
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1708__00MN = np.cross(v1704__00MH, v1707__00ML, axisa = 3, axisb = 3, axisc = 3)

# op _00Mo_power_combination_eval
# LANG: _00Mn --> eel_chord_length
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1693_eel_chord_length = (v1692__00Mn)
v1693_eel_chord_length = (v1693_eel_chord_length*_00Mo_coeff).reshape((1, 40, 4))

# op _00Ms_linear_combination_eval
# LANG: _00Mq, _00Mr --> _00Mt
# SHAPES: (1, 41, 4, 3), (1, 41, 4, 3) --> (1, 41, 4, 3)
# full namespace: MeshPreprocessing_comp
v1696__00Mt = v1694__00Mq+-1*v1695__00Mr

# op _00N7_linear_combination_eval
# LANG: _00N3, _00N6 --> _00N8
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1721__00N8 = v1718__00N3+v1720__00N6

# op _00N9_power_combination_eval
# LANG: _00M8 --> _00Na
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1722__00Na = (v1684__00M8)
v1722__00Na = (v1722__00Na*_00N9_coeff).reshape((1, 40, 4, 3))

# op _00NC_power_combination_eval
# LANG: v_inf_sq --> _00ND
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v1737__00ND = (v1734_v_inf_sq**0.5)
v1737__00ND = v1737__00ND.reshape((1, 1))

# op _00JC reshape_eval
# LANG: _00JB --> _00JD
# SHAPES: (1, 4) --> (1, 1, 4)
# full namespace: 
v1591__00JD = v1590__00JB.reshape((1, 1, 4))

# op _00JE_decompose_eval
# LANG: eel_gamma_w --> _00JM, _00JF, _00JL
# SHAPES: (1, 69, 4) --> (1, 68, 4), (1, 1, 4), (1, 68, 4)
# full namespace: 
v1592__00JF = ((v1743_eel_gamma_w.flatten())[src_indices__00JF__00JE]).reshape((1, 1, 4))
v1595__00JL = ((v1743_eel_gamma_w.flatten())[src_indices__00JL__00JE]).reshape((1, 68, 4))
v1596__00JM = ((v1743_eel_gamma_w.flatten())[src_indices__00JM__00JE]).reshape((1, 68, 4))

# op _00K4_linear_combination_eval
# LANG: _00JY, _00K3 --> _00K5
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1608__00K5 = v1604__00JY+v1607__00K3

# op _00KF_linear_combination_eval
# LANG: _00KD, _00KE --> _00KG
# SHAPES: (1, 68, 5, 3), (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v1627__00KG = v1625__00KD+-1*v1626__00KE

# op _00KI_power_combination_eval
# LANG: _00KH --> _00KJ
# SHAPES: (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v1629__00KJ = (v1628__00KH)
v1629__00KJ = (v1629__00KJ*_00KI_coeff).reshape((1, 68, 5, 3))

# op _00MO_power_combination_eval
# LANG: _00MN --> _00MP
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1709__00MP = (v1708__00MN**2)
v1709__00MP = v1709__00MP.reshape((1, 40, 4, 3))

# op _00Mu pnorm_axis_eval
# LANG: _00Mt --> _00Mv
# SHAPES: (1, 41, 4, 3) --> (1, 41, 4)
# full namespace: MeshPreprocessing_comp
v1697__00Mv = np.sum(v1696__00Mt**2,axis=(3,))**(1 / 2)

# op _00NE_power_combination_eval
# LANG: density, _00ND --> _00NF
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v1738__00NF = (v1733_density)*(v1737__00ND)
v1738__00NF = v1738__00NF.reshape((1, 1))

# op _00Nb_linear_combination_eval
# LANG: _00N8, _00Na --> _00Nc
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1723__00Nc = v1721__00N8+v1722__00Na

# op _00Ni_power_combination_eval
# LANG: _00MX --> _00Nj
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1726__00Nj = (v1714__00MX)
v1726__00Nj = (v1726__00Nj*_00Ni_coeff).reshape((1, 40, 4, 3))

# op _00Nk_power_combination_eval
# LANG: _00N4 --> _00Nl
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1727__00Nl = (v1719__00N4)
v1727__00Nl = (v1727__00Nl*_00Nk_coeff).reshape((1, 40, 4, 3))

# op _00Ny_single_tensor_sum_with_axis_eval
# LANG: eel_chord_length --> _00Nz
# SHAPES: (1, 40, 4) --> (1, 4)
# full namespace: MeshPreprocessing_comp
v1735__00Nz = np.sum(v1693_eel_chord_length, axis = (1,)).reshape((1, 4))

# op _00JG_linear_combination_eval
# LANG: _00JD, _00JF --> _00JH
# SHAPES: (1, 1, 4), (1, 1, 4) --> (1, 1, 4)
# full namespace: 
v1593__00JH = v1591__00JD+-1*v1592__00JF

# op _00JN_linear_combination_eval
# LANG: _00JL, _00JM --> _00JO
# SHAPES: (1, 68, 4), (1, 68, 4) --> (1, 68, 4)
# full namespace: 
v1597__00JO = v1595__00JL+-1*v1596__00JM

# op _00K8_linear_combination_eval
# LANG: _00K5, _00K7 --> _00K9
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1610__00K9 = v1608__00K5+-1*v1609__00K7

# op _00KK_linear_combination_eval
# LANG: _00KG, _00KJ --> _00KL
# SHAPES: (1, 68, 5, 3), (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v1630__00KL = v1627__00KG+v1629__00KJ

# op _00MQ_single_tensor_sum_with_axis_eval
# LANG: _00MP --> _00MR
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1710__00MR = np.sum(v1709__00MP, axis = (3,)).reshape((1, 40, 4))

# op _00Mw_decompose_eval
# LANG: _00Mv --> _00My, _00Mx
# SHAPES: (1, 41, 4) --> (1, 40, 4), (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1698__00Mx = ((v1697__00Mv.flatten())[src_indices__00Mx__00Mw]).reshape((1, 40, 4))
v1699__00My = ((v1697__00Mv.flatten())[src_indices__00My__00Mw]).reshape((1, 40, 4))

# op _00NA reshape_eval
# LANG: _00Nz --> _00NB
# SHAPES: (1, 4) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v1736__00NB = v1735__00Nz.reshape((1, 4, 1))

# op _00NG expand_array_eval
# LANG: _00NF --> _00NH
# SHAPES: (1, 1) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v1739__00NH = np.einsum('ac,b->abc', v1738__00NF.reshape((1, 1)) ,np.ones((4,))).reshape((1, 4, 1))

# op _00Nd reshape_eval
# LANG: _00Nc --> _00Ne
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: MeshPreprocessing_comp
v1724__00Ne = v1723__00Nc.reshape((1, 160, 3))

# op _00Nm_linear_combination_eval
# LANG: _00Nj, _00Nl --> _00Nn
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1728__00Nn = v1726__00Nj+v1727__00Nl

# op _00No_power_combination_eval
# LANG: _00M_ --> _00Np
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1729__00Np = (v1716__00M_)
v1729__00Np = (v1729__00Np*_00No_coeff).reshape((1, 40, 4, 3))

# op _00JI_power_combination_eval
# LANG: _00JH --> _00JJ
# SHAPES: (1, 1, 4) --> (1, 1, 4)
# full namespace: 
v1594__00JJ = (v1593__00JH)
v1594__00JJ = (v1594__00JJ*_00JI_coeff).reshape((1, 1, 4))

# op _00JP_power_combination_eval
# LANG: _00JO --> _00JQ
# SHAPES: (1, 68, 4) --> (1, 68, 4)
# full namespace: 
v1598__00JQ = (v1597__00JO)
v1598__00JQ = (v1598__00JQ*_00JP_coeff).reshape((1, 68, 4))

# op _00KM_power_combination_eval
# LANG: _00KL --> _00KN
# SHAPES: (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v1631__00KN = (v1630__00KL)
v1631__00KN = (v1631__00KN*_00KM_coeff).reshape((1, 68, 5, 3))

# op _00Ka_power_combination_eval
# LANG: _00K9 --> _00Kb
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v1611__00Kb = (v1610__00K9)
v1611__00Kb = (v1611__00Kb*_00Ka_coeff).reshape((1, 1, 5, 3))

# op _00MS_power_combination_eval
# LANG: _00MR --> _00MT
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1711__00MT = (v1710__00MR**0.5)
v1711__00MT = v1711__00MT.reshape((1, 40, 4))

# op _00Mz_linear_combination_eval
# LANG: _00Mx, _00My --> _00MA
# SHAPES: (1, 40, 4), (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1700__00MA = v1698__00Mx+v1699__00My

# op _00NI_power_combination_eval
# LANG: _00NH, _00NB --> _00NJ
# SHAPES: (1, 4, 1), (1, 4, 1) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v1740__00NJ = (v1739__00NH)*(v1736__00NB)
v1740__00NJ = v1740__00NJ.reshape((1, 4, 1))

# op _00Nf_linear_combination_eval
# LANG: _00Ne --> _00Ng
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: MeshPreprocessing_comp
v1725__00Ng = -1*v1724__00Ne

# op _00Nq_linear_combination_eval
# LANG: _00Nn, _00Np --> _00Nr
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1730__00Nr = v1728__00Nn+v1729__00Np

# op _00Ns_power_combination_eval
# LANG: _00M8 --> _00Nt
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1731__00Nt = (v1684__00M8)
v1731__00Nt = (v1731__00Nt*_00Ns_coeff).reshape((1, 40, 4, 3))

# op _00JK_indexed_passthrough_eval
# LANG: _00JJ, _00JQ --> eel_dgammaw_dt
# SHAPES: (1, 1, 4), (1, 68, 4) --> (1, 69, 4)
# full namespace: 
v1589_eel_dgammaw_dt__temp[i_v1594__00JJ__00JK_indexed_passthrough_eval] = v1594__00JJ.flatten()
v1589_eel_dgammaw_dt = v1589_eel_dgammaw_dt__temp.copy()
v1589_eel_dgammaw_dt__temp[i_v1598__00JQ__00JK_indexed_passthrough_eval] = v1598__00JQ.flatten()
v1589_eel_dgammaw_dt = v1589_eel_dgammaw_dt__temp.copy()

# op _00Kc_indexed_passthrough_eval
# LANG: _00Kb, _00KN --> eel_dwake_coords_dt
# SHAPES: (1, 1, 5, 3), (1, 68, 5, 3) --> (1, 69, 5, 3)
# full namespace: 
v1603_eel_dwake_coords_dt__temp[i_v1611__00Kb__00Kc_indexed_passthrough_eval] = v1611__00Kb.flatten()
v1603_eel_dwake_coords_dt = v1603_eel_dwake_coords_dt__temp.copy()
v1603_eel_dwake_coords_dt__temp[i_v1631__00KN__00Kc_indexed_passthrough_eval] = v1631__00KN.flatten()
v1603_eel_dwake_coords_dt = v1603_eel_dwake_coords_dt__temp.copy()

# op _00Lj_linear_combination_eval
# LANG: theta, gamma --> alpha
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1653_alpha = v1638_theta+-1*v1640_gamma

# op _00Ll_linear_combination_eval
# LANG: psi, psiw --> beta
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v1654_beta = v1639_psi+v1641_psiw

# op _00MB_power_combination_eval
# LANG: _00MA --> eel_span_length
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1701_eel_span_length = (v1700__00MA)
v1701_eel_span_length = (v1701_eel_span_length*_00MB_coeff).reshape((1, 40, 4))

# op _00MU_power_combination_eval
# LANG: _00MT --> eel_s_panel
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v1712_eel_s_panel = (v1711__00MT)
v1712_eel_s_panel = (v1712_eel_s_panel*_00MU_coeff).reshape((1, 40, 4))

# op _00NK_power_combination_eval
# LANG: _00NJ --> eel_re_span
# SHAPES: (1, 4, 1) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v1741_eel_re_span = (v1740__00NJ)
v1741_eel_re_span = (v1741_eel_re_span*_00NK_coeff).reshape((1, 4, 1))

# op _00Nh_indexed_passthrough_eval
# LANG: _00Ng --> bd_vec
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: MeshPreprocessing_comp
v1713_bd_vec__temp[i_v1725__00Ng__00Nh_indexed_passthrough_eval] = v1725__00Ng.flatten()
v1713_bd_vec = v1713_bd_vec__temp.copy()

# op _00Nu_linear_combination_eval
# LANG: _00Nr, _00Nt --> eel_eval_pts_coords
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v1732_eel_eval_pts_coords = v1730__00Nr+v1731__00Nt

# op _00Po_indexed_passthrough_eval
# LANG: _00Pl --> normal_concatenated_b
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v1798_normal_concatenated_b__temp[i_v1801__00Pl__00Po_indexed_passthrough_eval] = v1801__00Pl.flatten()
v1798_normal_concatenated_b = v1798_normal_concatenated_b__temp.copy()