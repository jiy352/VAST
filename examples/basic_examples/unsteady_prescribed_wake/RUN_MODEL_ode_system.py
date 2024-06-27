

# RUN_MODEL_ode_system

# system evaluation block

# op _002w_indexed_passthrough_eval
# LANG: u, w --> frame_vel
# SHAPES: (1, 1), (1, 1) --> (1, 3)
# full namespace: adapter_comp
v542_frame_vel__temp[i_v80_u__002w_indexed_passthrough_eval] = v80_u.flatten()
v542_frame_vel = v542_frame_vel__temp.copy()
v542_frame_vel__temp[i_v114_w__002w_indexed_passthrough_eval] = v114_w.flatten()
v542_frame_vel = v542_frame_vel__temp.copy()

# op _002P_decompose_eval
# LANG: frame_vel --> _002U, _002Q
# SHAPES: (1, 3) --> (1, 1), (1, 1)
# full namespace: MeshPreprocessing_comp
v116__002Q = ((v542_frame_vel.flatten())[src_indices__002Q__002P]).reshape((1, 1))
v118__002U = ((v542_frame_vel.flatten())[src_indices__002U__002P]).reshape((1, 1))

# op _002R_linear_combination_eval
# LANG: _002Q --> _002S
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v117__002S = -1*v116__002Q

# op _002V_linear_combination_eval
# LANG: _002U --> _002W
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v119__002W = -1*v118__002U

# op _002C_decompose_eval
# LANG: wing --> _003b, _002D, _002G, _0034, _0035, _003a, _003t, _003u, _003_, _0042, _0047
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 2, 2, 3), (1, 2, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v108__002D = ((v230_wing.flatten())[src_indices__002D__002C]).reshape((1, 1, 3, 3))
v110__002G = ((v230_wing.flatten())[src_indices__002G__002C]).reshape((1, 1, 3, 3))
v124__0034 = ((v230_wing.flatten())[src_indices__0034__002C]).reshape((1, 1, 2, 3))
v125__0035 = ((v230_wing.flatten())[src_indices__0035__002C]).reshape((1, 1, 2, 3))
v128__003a = ((v230_wing.flatten())[src_indices__003a__002C]).reshape((1, 1, 2, 3))
v129__003b = ((v230_wing.flatten())[src_indices__003b__002C]).reshape((1, 1, 2, 3))
v139__003t = ((v230_wing.flatten())[src_indices__003t__002C]).reshape((1, 2, 2, 3))
v140__003u = ((v230_wing.flatten())[src_indices__003u__002C]).reshape((1, 2, 2, 3))
v159__003_ = ((v230_wing.flatten())[src_indices__003___002C]).reshape((1, 1, 2, 3))
v161__0042 = ((v230_wing.flatten())[src_indices__0042__002C]).reshape((1, 1, 2, 3))
v164__0047 = ((v230_wing.flatten())[src_indices__0047__002C]).reshape((1, 1, 2, 3))

# op _002T_indexed_passthrough_eval
# LANG: _002S, _002W, w --> fs
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v115_fs__temp[i_v117__002S__002T_indexed_passthrough_eval] = v117__002S.flatten()
v115_fs = v115_fs__temp.copy()
v115_fs__temp[i_v119__002W__002T_indexed_passthrough_eval] = v119__002W.flatten()
v115_fs = v115_fs__temp.copy()
v115_fs__temp[i_v114_w__002T_indexed_passthrough_eval] = v114_w.flatten()
v115_fs = v115_fs__temp.copy()

# op _005m_decompose_eval
# LANG: wing_wake_coords --> _005n
# SHAPES: (1, 3, 3, 3) --> (1, 1, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v208__005n = ((v206_wing_wake_coords.flatten())[src_indices__005n__005m]).reshape((1, 1, 3, 3))

# op _002X_power_combination_eval
# LANG: fs --> _002Y
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v120__002Y = (v115_fs)
v120__002Y = (v120__002Y*_002X_coeff).reshape((1, 3))

# op _0036_linear_combination_eval
# LANG: _0034, _0035 --> _0037
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v126__0037 = v124__0034+v125__0035

# op _003c_linear_combination_eval
# LANG: _003b, _003a --> _003d
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v130__003d = v128__003a+v129__003b

# op _005o_power_combination_eval
# LANG: _005n --> _005p
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v209__005p = (v208__005n)
v209__005p = v209__005p.reshape((1, 1, 3, 3))

# op _002Z_power_combination_eval
# LANG: _002Y --> _002_
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v121__002_ = (v120__002Y)
v121__002_ = (v121__002_*_002Z_coeff).reshape((1, 3))

# op _0038_power_combination_eval
# LANG: _0037 --> _0039
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v127__0039 = (v126__0037)
v127__0039 = (v127__0039*_0038_coeff).reshape((1, 1, 2, 3))

# op _003e_power_combination_eval
# LANG: _003d --> _003f
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v131__003f = (v130__003d)
v131__003f = (v131__003f*_003e_coeff).reshape((1, 1, 2, 3))

# op _005l_indexed_passthrough_eval
# LANG: _005p, wing_wake_coords --> wing_TE_wake_coords
# SHAPES: (1, 1, 3, 3), (1, 3, 3, 3) --> (1, 4, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v254_wing_TE_wake_coords__temp[i_v206_wing_wake_coords__005l_indexed_passthrough_eval] = v206_wing_wake_coords.flatten()
v254_wing_TE_wake_coords = v254_wing_TE_wake_coords__temp.copy()
v254_wing_TE_wake_coords__temp[i_v209__005p__005l_indexed_passthrough_eval] = v209__005p.flatten()
v254_wing_TE_wake_coords = v254_wing_TE_wake_coords__temp.copy()

# op _002E_power_combination_eval
# LANG: _002D --> _002F
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v109__002F = (v108__002D)
v109__002F = (v109__002F*_002E_coeff).reshape((1, 1, 3, 3))

# op _002H_power_combination_eval
# LANG: _002G --> _002I
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v111__002I = (v110__002G)
v111__002I = (v111__002I*_002H_coeff).reshape((1, 1, 3, 3))

# op _0030 expand_array_eval
# LANG: _002_ --> _0031
# SHAPES: (1, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v122__0031 = np.einsum('ad,bc->abcd', v121__002_.reshape((1, 3)) ,np.ones((1, 3))).reshape((1, 1, 3, 3))

# op _003g_linear_combination_eval
# LANG: _0039, _003f --> wing_coll_pts_coords
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v432_wing_coll_pts_coords = v127__0039+v131__003f

# op _006C_decompose_eval
# LANG: wing_TE_wake_coords --> _006D, _006E, _006F, _006G
# SHAPES: (1, 4, 3, 3) --> (1, 3, 2, 3), (1, 3, 2, 3), (1, 3, 2, 3), (1, 3, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v255__006D = ((v254_wing_TE_wake_coords.flatten())[src_indices__006D__006C]).reshape((1, 3, 2, 3))
v256__006E = ((v254_wing_TE_wake_coords.flatten())[src_indices__006E__006C]).reshape((1, 3, 2, 3))
v257__006F = ((v254_wing_TE_wake_coords.flatten())[src_indices__006F__006C]).reshape((1, 3, 2, 3))
v258__006G = ((v254_wing_TE_wake_coords.flatten())[src_indices__006G__006C]).reshape((1, 3, 2, 3))

# op _002J_linear_combination_eval
# LANG: _002F, _002I --> _002K
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v112__002K = v109__002F+v111__002I

# op _0032_linear_combination_eval
# LANG: _002G, _0031 --> _0033
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v123__0033 = v110__002G+v122__0031

# op _006H reshape_eval
# LANG: wing_coll_pts_coords --> _006I
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v259__006I = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _006N reshape_eval
# LANG: _006D --> _006O
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v262__006O = v255__006D.reshape((1, 6, 3))

# op _0070 reshape_eval
# LANG: wing_coll_pts_coords --> _0071
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v269__0071 = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _0076 reshape_eval
# LANG: _006E --> _0077
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v272__0077 = v256__006E.reshape((1, 6, 3))

# op _007k reshape_eval
# LANG: wing_coll_pts_coords --> _007l
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v279__007l = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007q reshape_eval
# LANG: _006F --> _007r
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v282__007r = v257__006F.reshape((1, 6, 3))

# op _002L_indexed_passthrough_eval
# LANG: _002K, _0033 --> wing_bd_vtx_coords
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 2, 3, 3)
# full namespace: MeshPreprocessing_comp
v433_wing_bd_vtx_coords__temp[i_v112__002K__002L_indexed_passthrough_eval] = v112__002K.flatten()
v433_wing_bd_vtx_coords = v433_wing_bd_vtx_coords__temp.copy()
v433_wing_bd_vtx_coords__temp[i_v123__0033__002L_indexed_passthrough_eval] = v123__0033.flatten()
v433_wing_bd_vtx_coords = v433_wing_bd_vtx_coords__temp.copy()

# op _006J expand_array_eval
# LANG: _006I --> _006K
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v260__006K = np.einsum('abd,c->abcd', v259__006I.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _006P expand_array_eval
# LANG: _006O --> _006Q
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v263__006Q = np.einsum('acd,b->abcd', v262__006O.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _0072 expand_array_eval
# LANG: _0071 --> _0073
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v270__0073 = np.einsum('abd,c->abcd', v269__0071.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _0078 expand_array_eval
# LANG: _0077 --> _0079
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v273__0079 = np.einsum('acd,b->abcd', v272__0077.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _007E reshape_eval
# LANG: wing_coll_pts_coords --> _007F
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v289__007F = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007K reshape_eval
# LANG: _006G --> _007L
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v292__007L = v258__006G.reshape((1, 6, 3))

# op _007m expand_array_eval
# LANG: _007l --> _007n
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v280__007n = np.einsum('abd,c->abcd', v279__007l.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007s expand_array_eval
# LANG: _007r --> _007t
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v283__007t = np.einsum('acd,b->abcd', v282__007r.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _006L reshape_eval
# LANG: _006K --> _006M
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v261__006M = v260__006K.reshape((1, 12, 3))

# op _006R reshape_eval
# LANG: _006Q --> _006S
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v264__006S = v263__006Q.reshape((1, 12, 3))

# op _0074 reshape_eval
# LANG: _0073 --> _0075
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v271__0075 = v270__0073.reshape((1, 12, 3))

# op _007G expand_array_eval
# LANG: _007F --> _007H
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v290__007H = np.einsum('abd,c->abcd', v289__007F.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007M expand_array_eval
# LANG: _007L --> _007N
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v293__007N = np.einsum('acd,b->abcd', v292__007L.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _007a reshape_eval
# LANG: _0079 --> _007b
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v274__007b = v273__0079.reshape((1, 12, 3))

# op _007o reshape_eval
# LANG: _007n --> _007p
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v281__007p = v280__007n.reshape((1, 12, 3))

# op _007u reshape_eval
# LANG: _007t --> _007v
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v284__007v = v283__007t.reshape((1, 12, 3))

# op _00c9_decompose_eval
# LANG: wing_bd_vtx_coords --> _00ca, _00cb, _00cc, _00cd
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v434__00ca = ((v433_wing_bd_vtx_coords.flatten())[src_indices__00ca__00c9]).reshape((1, 1, 2, 3))
v435__00cb = ((v433_wing_bd_vtx_coords.flatten())[src_indices__00cb__00c9]).reshape((1, 1, 2, 3))
v436__00cc = ((v433_wing_bd_vtx_coords.flatten())[src_indices__00cc__00c9]).reshape((1, 1, 2, 3))
v437__00cd = ((v433_wing_bd_vtx_coords.flatten())[src_indices__00cd__00c9]).reshape((1, 1, 2, 3))

# op _006T_linear_combination_eval
# LANG: _006M, _006S --> _006U
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v265__006U = v261__006M+-1*v264__006S

# op _007I reshape_eval
# LANG: _007H --> _007J
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v291__007J = v290__007H.reshape((1, 12, 3))

# op _007O reshape_eval
# LANG: _007N --> _007P
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v294__007P = v293__007N.reshape((1, 12, 3))

# op _007c_linear_combination_eval
# LANG: _0075, _007b --> _007d
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v275__007d = v271__0075+-1*v274__007b

# op _007w_linear_combination_eval
# LANG: _007p, _007v --> _007x
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v285__007x = v281__007p+-1*v284__007v

# op _00cE reshape_eval
# LANG: _00cb --> _00cF
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v451__00cF = v435__00cb.reshape((1, 2, 3))

# op _00cS reshape_eval
# LANG: wing_coll_pts_coords --> _00cT
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v458__00cT = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00cY reshape_eval
# LANG: _00cc --> _00cZ
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v461__00cZ = v436__00cc.reshape((1, 2, 3))

# op _00ce reshape_eval
# LANG: wing_coll_pts_coords --> _00cf
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v438__00cf = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00ck reshape_eval
# LANG: _00ca --> _00cl
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v441__00cl = v434__00ca.reshape((1, 2, 3))

# op _00cy reshape_eval
# LANG: wing_coll_pts_coords --> _00cz
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v448__00cz = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _006V_power_combination_eval
# LANG: _006U --> _006W
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v266__006W = (v265__006U**2)
v266__006W = v266__006W.reshape((1, 12, 3))

# op _007Q_linear_combination_eval
# LANG: _007J, _007P --> _007R
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v295__007R = v291__007J+-1*v294__007P

# op _007e_power_combination_eval
# LANG: _007d --> _007f
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v276__007f = (v275__007d**2)
v276__007f = v276__007f.reshape((1, 12, 3))

# op _007y_power_combination_eval
# LANG: _007x --> _007z
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v286__007z = (v285__007x**2)
v286__007z = v286__007z.reshape((1, 12, 3))

# op _00cA expand_array_eval
# LANG: _00cz --> _00cB
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v449__00cB = np.einsum('abd,c->abcd', v448__00cz.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cG expand_array_eval
# LANG: _00cF --> _00cH
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v452__00cH = np.einsum('acd,b->abcd', v451__00cF.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cU expand_array_eval
# LANG: _00cT --> _00cV
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v459__00cV = np.einsum('abd,c->abcd', v458__00cT.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00c_ expand_array_eval
# LANG: _00cZ --> _00d0
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v462__00d0 = np.einsum('acd,b->abcd', v461__00cZ.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cg expand_array_eval
# LANG: _00cf --> _00ch
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v439__00ch = np.einsum('abd,c->abcd', v438__00cf.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cm expand_array_eval
# LANG: _00cl --> _00cn
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v442__00cn = np.einsum('acd,b->abcd', v441__00cl.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00db reshape_eval
# LANG: wing_coll_pts_coords --> _00dc
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v468__00dc = v432_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00dh reshape_eval
# LANG: _00cd --> _00di
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v471__00di = v437__00cd.reshape((1, 2, 3))

# op _006X_single_tensor_sum_with_axis_eval
# LANG: _006W --> _006Y
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v267__006Y = np.sum(v266__006W, axis = (2,)).reshape((1, 12))

# op _007A_single_tensor_sum_with_axis_eval
# LANG: _007z --> _007B
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v287__007B = np.sum(v286__007z, axis = (2,)).reshape((1, 12))

# op _007S_power_combination_eval
# LANG: _007R --> _007T
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v296__007T = (v295__007R**2)
v296__007T = v296__007T.reshape((1, 12, 3))

# op _007g_single_tensor_sum_with_axis_eval
# LANG: _007f --> _007h
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v277__007h = np.sum(v276__007f, axis = (2,)).reshape((1, 12))

# op _00cC reshape_eval
# LANG: _00cB --> _00cD
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v450__00cD = v449__00cB.reshape((1, 4, 3))

# op _00cI reshape_eval
# LANG: _00cH --> _00cJ
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v453__00cJ = v452__00cH.reshape((1, 4, 3))

# op _00cW reshape_eval
# LANG: _00cV --> _00cX
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v460__00cX = v459__00cV.reshape((1, 4, 3))

# op _00ci reshape_eval
# LANG: _00ch --> _00cj
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v440__00cj = v439__00ch.reshape((1, 4, 3))

# op _00co reshape_eval
# LANG: _00cn --> _00cp
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v443__00cp = v442__00cn.reshape((1, 4, 3))

# op _00d1 reshape_eval
# LANG: _00d0 --> _00d2
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v463__00d2 = v462__00d0.reshape((1, 4, 3))

# op _00dd expand_array_eval
# LANG: _00dc --> _00de
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v469__00de = np.einsum('abd,c->abcd', v468__00dc.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00dj expand_array_eval
# LANG: _00di --> _00dk
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v472__00dk = np.einsum('acd,b->abcd', v471__00di.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _006Z_power_combination_eval
# LANG: _006Y --> _006_
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v268__006_ = (v267__006Y**0.5)
v268__006_ = v268__006_.reshape((1, 12))

# op _007C_power_combination_eval
# LANG: _007B --> _007D
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v288__007D = (v287__007B**0.5)
v288__007D = v288__007D.reshape((1, 12))

# op _007U_single_tensor_sum_with_axis_eval
# LANG: _007T --> _007V
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v297__007V = np.sum(v296__007T, axis = (2,)).reshape((1, 12))

# op _007i_power_combination_eval
# LANG: _007h --> _007j
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v278__007j = (v277__007h**0.5)
v278__007j = v278__007j.reshape((1, 12))

# op _00cK_linear_combination_eval
# LANG: _00cD, _00cJ --> _00cL
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v454__00cL = v450__00cD+-1*v453__00cJ

# op _00cq_linear_combination_eval
# LANG: _00cj, _00cp --> _00cr
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v444__00cr = v440__00cj+-1*v443__00cp

# op _00d3_linear_combination_eval
# LANG: _00cX, _00d2 --> _00d4
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v464__00d4 = v460__00cX+-1*v463__00d2

# op _00df reshape_eval
# LANG: _00de --> _00dg
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v470__00dg = v469__00de.reshape((1, 4, 3))

# op _00dl reshape_eval
# LANG: _00dk --> _00dm
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v473__00dm = v472__00dk.reshape((1, 4, 3))

# op _007W_power_combination_eval
# LANG: _007V --> _007X
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v298__007X = (v297__007V**0.5)
v298__007X = v298__007X.reshape((1, 12))

# op _0081_power_combination_eval
# LANG: _006U, _007d --> _0082
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v301__0082 = (v265__006U)*(v275__007d)
v301__0082 = v301__0082.reshape((1, 12, 3))

# op _0085_power_combination_eval
# LANG: _006_ --> _0086
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v303__0086 = (v268__006_**2)
v303__0086 = v303__0086.reshape((1, 12))

# op _0087_power_combination_eval
# LANG: _007j --> _0088
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v304__0088 = (v278__007j**2)
v304__0088 = v304__0088.reshape((1, 12))

# op _008D_power_combination_eval
# LANG: _006_ --> _008E
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v320__008E = (v268__006_)
v320__008E = (v320__008E*_008D_coeff).reshape((1, 12))

# op _008Z_power_combination_eval
# LANG: _007x, _007d --> _008_
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v331__008_ = (v275__007d)*(v285__007x)
v331__008_ = v331__008_.reshape((1, 12, 3))

# op _0092_power_combination_eval
# LANG: _007j --> _0093
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v333__0093 = (v278__007j**2)
v333__0093 = v333__0093.reshape((1, 12))

# op _0094_power_combination_eval
# LANG: _007D --> _0095
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v334__0095 = (v288__007D**2)
v334__0095 = v334__0095.reshape((1, 12))

# op _009A_power_combination_eval
# LANG: _007j --> _009B
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v350__009B = (v278__007j)
v350__009B = (v350__009B*_009A_coeff).reshape((1, 12))

# op _00cM_power_combination_eval
# LANG: _00cL --> _00cN
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v455__00cN = (v454__00cL**2)
v455__00cN = v455__00cN.reshape((1, 4, 3))

# op _00cs_power_combination_eval
# LANG: _00cr --> _00ct
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v445__00ct = (v444__00cr**2)
v445__00ct = v445__00ct.reshape((1, 4, 3))

# op _00d5_power_combination_eval
# LANG: _00d4 --> _00d6
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v465__00d6 = (v464__00d4**2)
v465__00d6 = v465__00d6.reshape((1, 4, 3))

# op _00dn_linear_combination_eval
# LANG: _00dg, _00dm --> _00do
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v474__00do = v470__00dg+-1*v473__00dm

# op _0083_single_tensor_sum_with_axis_eval
# LANG: _0082 --> _0084
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v302__0084 = np.sum(v301__0082, axis = (2,)).reshape((1, 12))

# op _008B_linear_combination_eval
# LANG: _0086, _0088 --> _008C
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v319__008C = v303__0086+v304__0088

# op _008F_power_combination_eval
# LANG: _007j, _008E --> _008G
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v321__008G = (v320__008E)*(v278__007j)
v321__008G = v321__008G.reshape((1, 12))

# op _008b_linear_combination_eval
# LANG: _0086 --> _008c
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v306__008c = _008b_constant+v303__0086

# op _008l_linear_combination_eval
# LANG: _0088 --> _008m
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v311__008m = _008l_constant+v304__0088

# op _0090_single_tensor_sum_with_axis_eval
# LANG: _008_ --> _0091
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v332__0091 = np.sum(v331__008_, axis = (2,)).reshape((1, 12))

# op _0098_linear_combination_eval
# LANG: _0093 --> _0099
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v336__0099 = _0098_constant+v333__0093

# op _009C_power_combination_eval
# LANG: _007D, _009B --> _009D
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v351__009D = (v350__009B)*(v288__007D)
v351__009D = v351__009D.reshape((1, 12))

# op _009W_power_combination_eval
# LANG: _007R, _007x --> _009X
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v361__009X = (v285__007x)*(v295__007R)
v361__009X = v361__009X.reshape((1, 12, 3))

# op _009__power_combination_eval
# LANG: _007D --> _00a0
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v363__00a0 = (v288__007D**2)
v363__00a0 = v363__00a0.reshape((1, 12))

# op _009i_linear_combination_eval
# LANG: _0095 --> _009j
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v341__009j = _009i_constant+v334__0095

# op _009y_linear_combination_eval
# LANG: _0093, _0095 --> _009z
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v349__009z = v333__0093+v334__0095

# op _00a1_power_combination_eval
# LANG: _007X --> _00a2
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v364__00a2 = (v298__007X**2)
v364__00a2 = v364__00a2.reshape((1, 12))

# op _00ax_power_combination_eval
# LANG: _007D --> _00ay
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v380__00ay = (v288__007D)
v380__00ay = (v380__00ay*_00ax_coeff).reshape((1, 12))

# op _00cO_single_tensor_sum_with_axis_eval
# LANG: _00cN --> _00cP
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v456__00cP = np.sum(v455__00cN, axis = (2,)).reshape((1, 4))

# op _00cu_single_tensor_sum_with_axis_eval
# LANG: _00ct --> _00cv
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v446__00cv = np.sum(v445__00ct, axis = (2,)).reshape((1, 4))

# op _00d7_single_tensor_sum_with_axis_eval
# LANG: _00d6 --> _00d8
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v466__00d8 = np.sum(v465__00d6, axis = (2,)).reshape((1, 4))

# op _00dp_power_combination_eval
# LANG: _00do --> _00dq
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v475__00dq = (v474__00do**2)
v475__00dq = v475__00dq.reshape((1, 4, 3))

# op _008H_linear_combination_eval
# LANG: _008C, _008G --> _008I
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v322__008I = v319__008C+-1*v321__008G

# op _008d_linear_combination_eval
# LANG: _008c --> _008e
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v307__008e = _008d_constant+v306__008c

# op _008n_linear_combination_eval
# LANG: _008m --> _008o
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v312__008o = _008n_constant+v311__008m

# op _008v_power_combination_eval
# LANG: _0086, _0088 --> _008w
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v316__008w = (v303__0086)*(v304__0088)
v316__008w = v316__008w.reshape((1, 12))

# op _008x_power_combination_eval
# LANG: _0084 --> _008y
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v317__008y = (v302__0084**2)
v317__008y = v317__008y.reshape((1, 12))

# op _009E_linear_combination_eval
# LANG: _009z, _009D --> _009F
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v352__009F = v349__009z+-1*v351__009D

# op _009Y_single_tensor_sum_with_axis_eval
# LANG: _009X --> _009Z
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v362__009Z = np.sum(v361__009X, axis = (2,)).reshape((1, 12))

# op _009a_linear_combination_eval
# LANG: _0099 --> _009b
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v337__009b = _009a_constant+v336__0099

# op _009k_linear_combination_eval
# LANG: _009j --> _009l
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v342__009l = _009k_constant+v341__009j

# op _009s_power_combination_eval
# LANG: _0093, _0095 --> _009t
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v346__009t = (v333__0093)*(v334__0095)
v346__009t = v346__009t.reshape((1, 12))

# op _009u_power_combination_eval
# LANG: _0091 --> _009v
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v347__009v = (v332__0091**2)
v347__009v = v347__009v.reshape((1, 12))

# op _00a5_linear_combination_eval
# LANG: _00a0 --> _00a6
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v366__00a6 = _00a5_constant+v363__00a0

# op _00aT_power_combination_eval
# LANG: _007R, _006U --> _00aU
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v391__00aU = (v295__007R)*(v265__006U)
v391__00aU = v391__00aU.reshape((1, 12, 3))

# op _00aX_power_combination_eval
# LANG: _007X --> _00aY
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v393__00aY = (v298__007X**2)
v393__00aY = v393__00aY.reshape((1, 12))

# op _00aZ_power_combination_eval
# LANG: _006_ --> _00a_
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v394__00a_ = (v268__006_**2)
v394__00a_ = v394__00a_.reshape((1, 12))

# op _00af_linear_combination_eval
# LANG: _00a2 --> _00ag
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v371__00ag = _00af_constant+v364__00a2

# op _00av_linear_combination_eval
# LANG: _00a0, _00a2 --> _00aw
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v379__00aw = v363__00a0+v364__00a2

# op _00az_power_combination_eval
# LANG: _007X, _00ay --> _00aA
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v381__00aA = (v380__00ay)*(v298__007X)
v381__00aA = v381__00aA.reshape((1, 12))

# op _00bu_power_combination_eval
# LANG: _007X --> _00bv
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v410__00bv = (v298__007X)
v410__00bv = (v410__00bv*_00bu_coeff).reshape((1, 12))

# op _00cQ_power_combination_eval
# LANG: _00cP --> _00cR
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v457__00cR = (v456__00cP**0.5)
v457__00cR = v457__00cR.reshape((1, 4))

# op _00cw_power_combination_eval
# LANG: _00cv --> _00cx
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v447__00cx = (v446__00cv**0.5)
v447__00cx = v447__00cx.reshape((1, 4))

# op _00d9_power_combination_eval
# LANG: _00d8 --> _00da
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v467__00da = (v466__00d8**0.5)
v467__00da = v467__00da.reshape((1, 4))

# op _00dZ_power_combination_eval
# LANG: _00d4, _00cL --> _00d_
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v493__00d_ = (v454__00cL)*(v464__00d4)
v493__00d_ = v493__00d_.reshape((1, 4, 3))

# op _00dr_single_tensor_sum_with_axis_eval
# LANG: _00dq --> _00ds
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v476__00ds = np.sum(v475__00dq, axis = (2,)).reshape((1, 4))

# op _00dz_power_combination_eval
# LANG: _00cr, _00cL --> _00dA
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v480__00dA = (v444__00cr)*(v454__00cL)
v480__00dA = v480__00dA.reshape((1, 4, 3))

# op _0089_linear_combination_eval
# LANG: _0086, _0084 --> _008a
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v305__008a = v303__0086+-1*v302__0084

# op _008J_power_combination_eval
# LANG: _008I --> _008K
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v323__008K = (v322__008I)
v323__008K = (v323__008K*_008J_coeff).reshape((1, 12))

# op _008f_power_combination_eval
# LANG: _008e --> _008g
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v308__008g = (v307__008e**0.5)
v308__008g = v308__008g.reshape((1, 12))

# op _008j_linear_combination_eval
# LANG: _0088, _0084 --> _008k
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v310__008k = v304__0088+-1*v302__0084

# op _008p_power_combination_eval
# LANG: _008o --> _008q
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v313__008q = (v312__008o**0.5)
v313__008q = v313__008q.reshape((1, 12))

# op _008z_linear_combination_eval
# LANG: _008w, _008y --> _008A
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v318__008A = v316__008w+-1*v317__008y

# op _0096_linear_combination_eval
# LANG: _0093, _0091 --> _0097
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v335__0097 = v333__0093+-1*v332__0091

# op _009G_power_combination_eval
# LANG: _009F --> _009H
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v353__009H = (v352__009F)
v353__009H = (v353__009H*_009G_coeff).reshape((1, 12))

# op _009c_power_combination_eval
# LANG: _009b --> _009d
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v338__009d = (v337__009b**0.5)
v338__009d = v338__009d.reshape((1, 12))

# op _009g_linear_combination_eval
# LANG: _0095, _0091 --> _009h
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v340__009h = v334__0095+-1*v332__0091

# op _009m_power_combination_eval
# LANG: _009l --> _009n
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v343__009n = (v342__009l**0.5)
v343__009n = v343__009n.reshape((1, 12))

# op _009w_linear_combination_eval
# LANG: _009t, _009v --> _009x
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v348__009x = v346__009t+-1*v347__009v

# op _00a7_linear_combination_eval
# LANG: _00a6 --> _00a8
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v367__00a8 = _00a7_constant+v366__00a6

# op _00aB_linear_combination_eval
# LANG: _00aw, _00aA --> _00aC
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v382__00aC = v379__00aw+-1*v381__00aA

# op _00aV_single_tensor_sum_with_axis_eval
# LANG: _00aU --> _00aW
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v392__00aW = np.sum(v391__00aU, axis = (2,)).reshape((1, 12))

# op _00ah_linear_combination_eval
# LANG: _00ag --> _00ai
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v372__00ai = _00ah_constant+v371__00ag

# op _00ap_power_combination_eval
# LANG: _00a0, _00a2 --> _00aq
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v376__00aq = (v363__00a0)*(v364__00a2)
v376__00aq = v376__00aq.reshape((1, 12))

# op _00ar_power_combination_eval
# LANG: _009Z --> _00as
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v377__00as = (v362__009Z**2)
v377__00as = v377__00as.reshape((1, 12))

# op _00b2_linear_combination_eval
# LANG: _00aY --> _00b3
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v396__00b3 = _00b2_constant+v393__00aY

# op _00bc_linear_combination_eval
# LANG: _00a_ --> _00bd
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v401__00bd = _00bc_constant+v394__00a_

# op _00bs_linear_combination_eval
# LANG: _00aY, _00a_ --> _00bt
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v409__00bt = v393__00aY+v394__00a_

# op _00bw_power_combination_eval
# LANG: _00bv, _006_ --> _00bx
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v411__00bx = (v410__00bv)*(v268__006_)
v411__00bx = v411__00bx.reshape((1, 12))

# op _00dB_single_tensor_sum_with_axis_eval
# LANG: _00dA --> _00dC
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v481__00dC = np.sum(v480__00dA, axis = (2,)).reshape((1, 4))

# op _00dD_power_combination_eval
# LANG: _00cx, _00cR --> _00dE
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v482__00dE = (v447__00cx)*(v457__00cR)
v482__00dE = v482__00dE.reshape((1, 4))

# op _00dt_power_combination_eval
# LANG: _00ds --> _00du
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v477__00du = (v476__00ds**0.5)
v477__00du = v477__00du.reshape((1, 4))

# op _00e0_single_tensor_sum_with_axis_eval
# LANG: _00d_ --> _00e1
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v494__00e1 = np.sum(v493__00d_, axis = (2,)).reshape((1, 4))

# op _00e2_power_combination_eval
# LANG: _00da, _00cR --> _00e3
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v495__00e3 = (v457__00cR)*(v467__00da)
v495__00e3 = v495__00e3.reshape((1, 4))

# op _00eo_power_combination_eval
# LANG: _00do, _00d4 --> _00ep
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v506__00ep = (v464__00d4)*(v474__00do)
v506__00ep = v506__00ep.reshape((1, 4, 3))

# op _005__decompose_eval
# LANG: wing --> _0065, _0060, _0061, _0064
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v231__0060 = ((v230_wing.flatten())[src_indices__0060__005_]).reshape((1, 1, 2, 3))
v232__0061 = ((v230_wing.flatten())[src_indices__0061__005_]).reshape((1, 1, 2, 3))
v234__0064 = ((v230_wing.flatten())[src_indices__0064__005_]).reshape((1, 1, 2, 3))
v235__0065 = ((v230_wing.flatten())[src_indices__0065__005_]).reshape((1, 1, 2, 3))

# op _008L_linear_combination_eval
# LANG: _008A, _008K --> _008M
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v324__008M = v318__008A+v323__008K

# op _008h_power_combination_eval
# LANG: _008a, _008g --> _008i
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v309__008i = (v305__008a)*(v308__008g**-1)
v309__008i = v309__008i.reshape((1, 12))

# op _008r_power_combination_eval
# LANG: _008k, _008q --> _008s
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v314__008s = (v310__008k)*(v313__008q**-1)
v314__008s = v314__008s.reshape((1, 12))

# op _009I_linear_combination_eval
# LANG: _009x, _009H --> _009J
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v354__009J = v348__009x+v353__009H

# op _009e_power_combination_eval
# LANG: _0097, _009d --> _009f
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v339__009f = (v335__0097)*(v338__009d**-1)
v339__009f = v339__009f.reshape((1, 12))

# op _009o_power_combination_eval
# LANG: _009h, _009n --> _009p
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v344__009p = (v340__009h)*(v343__009n**-1)
v344__009p = v344__009p.reshape((1, 12))

# op _00a3_linear_combination_eval
# LANG: _00a0, _009Z --> _00a4
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v365__00a4 = v363__00a0+-1*v362__009Z

# op _00a9_power_combination_eval
# LANG: _00a8 --> _00aa
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v368__00aa = (v367__00a8**0.5)
v368__00aa = v368__00aa.reshape((1, 12))

# op _00aD_power_combination_eval
# LANG: _00aC --> _00aE
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v383__00aE = (v382__00aC)
v383__00aE = (v383__00aE*_00aD_coeff).reshape((1, 12))

# op _00ad_linear_combination_eval
# LANG: _00a2, _009Z --> _00ae
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v370__00ae = v364__00a2+-1*v362__009Z

# op _00aj_power_combination_eval
# LANG: _00ai --> _00ak
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v373__00ak = (v372__00ai**0.5)
v373__00ak = v373__00ak.reshape((1, 12))

# op _00at_linear_combination_eval
# LANG: _00aq, _00as --> _00au
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v378__00au = v376__00aq+-1*v377__00as

# op _00b4_linear_combination_eval
# LANG: _00b3 --> _00b5
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v397__00b5 = _00b4_constant+v396__00b3

# op _00be_linear_combination_eval
# LANG: _00bd --> _00bf
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v402__00bf = _00be_constant+v401__00bd

# op _00bm_power_combination_eval
# LANG: _00aY, _00a_ --> _00bn
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v406__00bn = (v393__00aY)*(v394__00a_)
v406__00bn = v406__00bn.reshape((1, 12))

# op _00bo_power_combination_eval
# LANG: _00aW --> _00bp
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v407__00bp = (v392__00aW**2)
v407__00bp = v407__00bp.reshape((1, 12))

# op _00by_linear_combination_eval
# LANG: _00bt, _00bx --> _00bz
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v412__00bz = v409__00bt+-1*v411__00bx

# op _00dF_linear_combination_eval
# LANG: _00dE, _00dC --> _00dG
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v483__00dG = v482__00dE+v481__00dC

# op _00dJ_power_combination_eval
# LANG: _00cx --> _00dK
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v485__00dK = (v447__00cx**-1)
v485__00dK = v485__00dK.reshape((1, 4))

# op _00dL_power_combination_eval
# LANG: _00cR --> _00dM
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v486__00dM = (v457__00cR**-1)
v486__00dM = v486__00dM.reshape((1, 4))

# op _00e4_linear_combination_eval
# LANG: _00e3, _00e1 --> _00e5
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v496__00e5 = v495__00e3+v494__00e1

# op _00e8_power_combination_eval
# LANG: _00cR --> _00e9
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v498__00e9 = (v457__00cR**-1)
v498__00e9 = v498__00e9.reshape((1, 4))

# op _00eO_power_combination_eval
# LANG: _00do, _00cr --> _00eP
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v519__00eP = (v474__00do)*(v444__00cr)
v519__00eP = v519__00eP.reshape((1, 4, 3))

# op _00ea_power_combination_eval
# LANG: _00da --> _00eb
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v499__00eb = (v467__00da**-1)
v499__00eb = v499__00eb.reshape((1, 4))

# op _00eq_single_tensor_sum_with_axis_eval
# LANG: _00ep --> _00er
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v507__00er = np.sum(v506__00ep, axis = (2,)).reshape((1, 4))

# op _00es_power_combination_eval
# LANG: _00du, _00da --> _00et
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v508__00et = (v467__00da)*(v477__00du)
v508__00et = v508__00et.reshape((1, 4))

# op _0062_linear_combination_eval
# LANG: _0060, _0061 --> _0063
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v233__0063 = v231__0060+-1*v232__0061

# op _0066_linear_combination_eval
# LANG: _0064, _0065 --> _0067
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v236__0067 = v234__0064+-1*v235__0065

# op _008N_linear_combination_eval
# LANG: _008M --> _008O
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v325__008O = _008N_constant+v324__008M

# op _008t_linear_combination_eval
# LANG: _008i, _008s --> _008u
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v315__008u = v309__008i+v314__008s

# op _009K_linear_combination_eval
# LANG: _009J --> _009L
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v355__009L = _009K_constant+v354__009J

# op _009q_linear_combination_eval
# LANG: _009f, _009p --> _009r
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v345__009r = v339__009f+v344__009p

# op _00aF_linear_combination_eval
# LANG: _00au, _00aE --> _00aG
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v384__00aG = v378__00au+v383__00aE

# op _00ab_power_combination_eval
# LANG: _00a4, _00aa --> _00ac
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v369__00ac = (v365__00a4)*(v368__00aa**-1)
v369__00ac = v369__00ac.reshape((1, 12))

# op _00al_power_combination_eval
# LANG: _00ae, _00ak --> _00am
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v374__00am = (v370__00ae)*(v373__00ak**-1)
v374__00am = v374__00am.reshape((1, 12))

# op _00b0_linear_combination_eval
# LANG: _00aY, _00aW --> _00b1
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v395__00b1 = v393__00aY+-1*v392__00aW

# op _00b6_power_combination_eval
# LANG: _00b5 --> _00b7
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v398__00b7 = (v397__00b5**0.5)
v398__00b7 = v398__00b7.reshape((1, 12))

# op _00bA_power_combination_eval
# LANG: _00bz --> _00bB
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v413__00bB = (v412__00bz)
v413__00bB = (v413__00bB*_00bA_coeff).reshape((1, 12))

# op _00ba_linear_combination_eval
# LANG: _00a_, _00aW --> _00bb
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v400__00bb = v394__00a_+-1*v392__00aW

# op _00bg_power_combination_eval
# LANG: _00bf --> _00bh
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v403__00bh = (v402__00bf**0.5)
v403__00bh = v403__00bh.reshape((1, 12))

# op _00bq_linear_combination_eval
# LANG: _00bn, _00bp --> _00br
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v408__00br = v406__00bn+-1*v407__00bp

# op _00dH_power_combination_eval
# LANG: _00dG --> _00dI
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v484__00dI = (v483__00dG**-1)
v484__00dI = v484__00dI.reshape((1, 4))

# op _00dN_linear_combination_eval
# LANG: _00dK, _00dM --> _00dO
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v487__00dO = v485__00dK+v486__00dM

# op _00e6_power_combination_eval
# LANG: _00e5 --> _00e7
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v497__00e7 = (v496__00e5**-1)
v497__00e7 = v497__00e7.reshape((1, 4))

# op _00eA_power_combination_eval
# LANG: _00du --> _00eB
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v512__00eB = (v477__00du**-1)
v512__00eB = v512__00eB.reshape((1, 4))

# op _00eQ_single_tensor_sum_with_axis_eval
# LANG: _00eP --> _00eR
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v520__00eR = np.sum(v519__00eP, axis = (2,)).reshape((1, 4))

# op _00eS_power_combination_eval
# LANG: _00cx, _00du --> _00eT
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v521__00eT = (v477__00du)*(v447__00cx)
v521__00eT = v521__00eT.reshape((1, 4))

# op _00ec_linear_combination_eval
# LANG: _00e9, _00eb --> _00ed
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v500__00ed = v498__00e9+v499__00eb

# op _00eu_linear_combination_eval
# LANG: _00et, _00er --> _00ev
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v509__00ev = v508__00et+v507__00er

# op _00ey_power_combination_eval
# LANG: _00da --> _00ez
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v511__00ez = (v467__00da**-1)
v511__00ez = v511__00ez.reshape((1, 4))

# op _0068 cross_product_eval
# LANG: _0063, _0067 --> _0069
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v237__0069 = np.cross(v233__0063, v236__0067, axisa = 3, axisb = 3, axisc = 3)

# op _007Y cross_product_eval
# LANG: _006U, _007d --> _007Z
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v299__007Z = np.cross(v265__006U, v275__007d, axisa = 2, axisb = 2, axisc = 2)

# op _008P_power_combination_eval
# LANG: _008u, _008O --> _008Q
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v326__008Q = (v315__008u)*(v325__008O**-1)
v326__008Q = v326__008Q.reshape((1, 12))

# op _008V cross_product_eval
# LANG: _007x, _007d --> _008W
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v329__008W = np.cross(v275__007d, v285__007x, axisa = 2, axisb = 2, axisc = 2)

# op _009M_power_combination_eval
# LANG: _009r, _009L --> _009N
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v356__009N = (v345__009r)*(v355__009L**-1)
v356__009N = v356__009N.reshape((1, 12))

# op _00aH_linear_combination_eval
# LANG: _00aG --> _00aI
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v385__00aI = _00aH_constant+v384__00aG

# op _00an_linear_combination_eval
# LANG: _00ac, _00am --> _00ao
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v375__00ao = v369__00ac+v374__00am

# op _00b8_power_combination_eval
# LANG: _00b1, _00b7 --> _00b9
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v399__00b9 = (v395__00b1)*(v398__00b7**-1)
v399__00b9 = v399__00b9.reshape((1, 12))

# op _00bC_linear_combination_eval
# LANG: _00br, _00bB --> _00bD
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v414__00bD = v408__00br+v413__00bB

# op _00bi_power_combination_eval
# LANG: _00bb, _00bh --> _00bj
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v404__00bj = (v400__00bb)*(v403__00bh**-1)
v404__00bj = v404__00bj.reshape((1, 12))

# op _00dP_power_combination_eval
# LANG: _00dI, _00dO --> _00dQ
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v488__00dQ = (v484__00dI)*(v487__00dO)
v488__00dQ = v488__00dQ.reshape((1, 4))

# op _00dV cross_product_eval
# LANG: _00d4, _00cL --> _00dW
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v491__00dW = np.cross(v454__00cL, v464__00d4, axisa = 2, axisb = 2, axisc = 2)

# op _00dv cross_product_eval
# LANG: _00cr, _00cL --> _00dw
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v478__00dw = np.cross(v444__00cr, v454__00cL, axisa = 2, axisb = 2, axisc = 2)

# op _00eC_linear_combination_eval
# LANG: _00ez, _00eB --> _00eD
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v513__00eD = v511__00ez+v512__00eB

# op _00eU_linear_combination_eval
# LANG: _00eT, _00eR --> _00eV
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v522__00eV = v521__00eT+v520__00eR

# op _00eY_power_combination_eval
# LANG: _00du --> _00eZ
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v524__00eZ = (v477__00du**-1)
v524__00eZ = v524__00eZ.reshape((1, 4))

# op _00e__power_combination_eval
# LANG: _00cx --> _00f0
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v525__00f0 = (v447__00cx**-1)
v525__00f0 = v525__00f0.reshape((1, 4))

# op _00ee_power_combination_eval
# LANG: _00e7, _00ed --> _00ef
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v501__00ef = (v497__00e7)*(v500__00ed)
v501__00ef = v501__00ef.reshape((1, 4))

# op _00ew_power_combination_eval
# LANG: _00ev --> _00ex
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v510__00ex = (v509__00ev**-1)
v510__00ex = v510__00ex.reshape((1, 4))

# op _005B_indexed_passthrough_eval
# LANG: p, q, r --> ang_vel
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v216_ang_vel__temp[i_v213_p__005B_indexed_passthrough_eval] = v213_p.flatten()
v216_ang_vel = v216_ang_vel__temp.copy()
v216_ang_vel__temp[i_v214_q__005B_indexed_passthrough_eval] = v214_q.flatten()
v216_ang_vel = v216_ang_vel__temp.copy()
v216_ang_vel__temp[i_v215_r__005B_indexed_passthrough_eval] = v215_r.flatten()
v216_ang_vel = v216_ang_vel__temp.copy()

# op _005E expand_array_eval
# LANG: wing_rot_ref --> _005F
# SHAPES: (1, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v219__005F = np.einsum('ad,bc->abcd', v218_wing_rot_ref.reshape((1, 3)) ,np.ones((1, 2))).reshape((1, 1, 2, 3))

# op _006a_power_combination_eval
# LANG: _0069 --> _006b
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v238__006b = (v237__0069**2)
v238__006b = v238__006b.reshape((1, 1, 2, 3))

# op _007__power_combination_eval
# LANG: _007Z --> _0080
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v300__0080 = (v299__007Z)
v300__0080 = (v300__0080*_007__coeff).reshape((1, 12, 3))

# op _008R expand_array_eval
# LANG: _008Q --> _008S
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v327__008S = np.einsum('ab,c->abc', v326__008Q.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _008X_power_combination_eval
# LANG: _008W --> _008Y
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v330__008Y = (v329__008W)
v330__008Y = (v330__008Y*_008X_coeff).reshape((1, 12, 3))

# op _009O expand_array_eval
# LANG: _009N --> _009P
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v357__009P = np.einsum('ab,c->abc', v356__009N.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _009S cross_product_eval
# LANG: _007R, _007x --> _009T
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v359__009T = np.cross(v285__007x, v295__007R, axisa = 2, axisb = 2, axisc = 2)

# op _00aJ_power_combination_eval
# LANG: _00ao, _00aI --> _00aK
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v386__00aK = (v375__00ao)*(v385__00aI**-1)
v386__00aK = v386__00aK.reshape((1, 12))

# op _00bE_linear_combination_eval
# LANG: _00bD --> _00bF
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v415__00bF = _00bE_constant+v414__00bD

# op _00bk_linear_combination_eval
# LANG: _00b9, _00bj --> _00bl
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v405__00bl = v399__00b9+v404__00bj

# op _00dR expand_array_eval
# LANG: _00dQ --> _00dS
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v489__00dS = np.einsum('ab,c->abc', v488__00dQ.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00dX_power_combination_eval
# LANG: _00dW --> _00dY
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v492__00dY = (v491__00dW)
v492__00dY = (v492__00dY*_00dX_coeff).reshape((1, 4, 3))

# op _00dx_power_combination_eval
# LANG: _00dw --> _00dy
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v479__00dy = (v478__00dw)
v479__00dy = (v479__00dy*_00dx_coeff).reshape((1, 4, 3))

# op _00eE_power_combination_eval
# LANG: _00ex, _00eD --> _00eF
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v514__00eF = (v510__00ex)*(v513__00eD)
v514__00eF = v514__00eF.reshape((1, 4))

# op _00eW_power_combination_eval
# LANG: _00eV --> _00eX
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v523__00eX = (v522__00eV**-1)
v523__00eX = v523__00eX.reshape((1, 4))

# op _00eg expand_array_eval
# LANG: _00ef --> _00eh
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v502__00eh = np.einsum('ab,c->abc', v501__00ef.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00ek cross_product_eval
# LANG: _00do, _00d4 --> _00el
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v504__00el = np.cross(v464__00d4, v474__00do, axisa = 2, axisb = 2, axisc = 2)

# op _00f1_linear_combination_eval
# LANG: _00eZ, _00f0 --> _00f2
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v526__00f2 = v524__00eZ+v525__00f0

# op _005G_linear_combination_eval
# LANG: _005F, wing_coll_pts_coords --> _005H
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v220__005H = v432_wing_coll_pts_coords+-1*v219__005F

# op _005I expand_array_eval
# LANG: ang_vel --> _005J
# SHAPES: (1, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v221__005J = np.einsum('ad,bc->abcd', v216_ang_vel.reshape((1, 3)) ,np.ones((1, 2))).reshape((1, 1, 2, 3))

# op _006c_single_tensor_sum_with_axis_eval
# LANG: _006b --> _006d
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v239__006d = np.sum(v238__006b, axis = (3,)).reshape((1, 1, 2))

# op _008T_power_combination_eval
# LANG: _008S, _0080 --> _008U
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v328__008U = (v327__008S)*(v300__0080)
v328__008U = v328__008U.reshape((1, 12, 3))

# op _009Q_power_combination_eval
# LANG: _009P, _008Y --> _009R
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v358__009R = (v357__009P)*(v330__008Y)
v358__009R = v358__009R.reshape((1, 12, 3))

# op _009U_power_combination_eval
# LANG: _009T --> _009V
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v360__009V = (v359__009T)
v360__009V = (v360__009V*_009U_coeff).reshape((1, 12, 3))

# op _00aL expand_array_eval
# LANG: _00aK --> _00aM
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v387__00aM = np.einsum('ab,c->abc', v386__00aK.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _00aP cross_product_eval
# LANG: _007R, _006U --> _00aQ
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v389__00aQ = np.cross(v295__007R, v265__006U, axisa = 2, axisb = 2, axisc = 2)

# op _00bG_power_combination_eval
# LANG: _00bl, _00bF --> _00bH
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v416__00bH = (v405__00bl)*(v415__00bF**-1)
v416__00bH = v416__00bH.reshape((1, 12))

# op _00dT_power_combination_eval
# LANG: _00dS, _00dy --> _00dU
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v490__00dU = (v489__00dS)*(v479__00dy)
v490__00dU = v490__00dU.reshape((1, 4, 3))

# op _00eG expand_array_eval
# LANG: _00eF --> _00eH
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v515__00eH = np.einsum('ab,c->abc', v514__00eF.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00eK cross_product_eval
# LANG: _00do, _00cr --> _00eL
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v517__00eL = np.cross(v474__00do, v444__00cr, axisa = 2, axisb = 2, axisc = 2)

# op _00ei_power_combination_eval
# LANG: _00eh, _00dY --> _00ej
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v503__00ej = (v502__00eh)*(v492__00dY)
v503__00ej = v503__00ej.reshape((1, 4, 3))

# op _00em_power_combination_eval
# LANG: _00el --> _00en
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v505__00en = (v504__00el)
v505__00en = (v505__00en*_00em_coeff).reshape((1, 4, 3))

# op _00f3_power_combination_eval
# LANG: _00eX, _00f2 --> _00f4
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v527__00f4 = (v523__00eX)*(v526__00f2)
v527__00f4 = v527__00f4.reshape((1, 4))

# op _005K cross_product_eval
# LANG: _005J, _005H --> _005L
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v222__005L = np.cross(v221__005J, v220__005H, axisa = 3, axisb = 3, axisc = 3)

# op _006e_power_combination_eval
# LANG: _006d --> _006f
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v240__006f = (v239__006d**0.5)
v240__006f = v240__006f.reshape((1, 1, 2))

# op _00aN_power_combination_eval
# LANG: _00aM, _009V --> _00aO
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v388__00aO = (v387__00aM)*(v360__009V)
v388__00aO = v388__00aO.reshape((1, 12, 3))

# op _00aR_power_combination_eval
# LANG: _00aQ --> _00aS
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v390__00aS = (v389__00aQ)
v390__00aS = (v390__00aS*_00aR_coeff).reshape((1, 12, 3))

# op _00bI expand_array_eval
# LANG: _00bH --> _00bJ
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v417__00bJ = np.einsum('ab,c->abc', v416__00bH.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _00bM_linear_combination_eval
# LANG: _008U, _009R --> _00bN
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v419__00bN = v328__008U+v358__009R

# op _00eI_power_combination_eval
# LANG: _00eH, _00en --> _00eJ
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v516__00eJ = (v515__00eH)*(v505__00en)
v516__00eJ = v516__00eJ.reshape((1, 4, 3))

# op _00eM_power_combination_eval
# LANG: _00eL --> _00eN
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v518__00eN = (v517__00eL)
v518__00eN = (v518__00eN*_00eM_coeff).reshape((1, 4, 3))

# op _00f5 expand_array_eval
# LANG: _00f4 --> _00f6
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v528__00f6 = np.einsum('ab,c->abc', v527__00f4.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00f9_linear_combination_eval
# LANG: _00dU, _00ej --> _00fa
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v530__00fa = v490__00dU+v503__00ej

# op _005M reshape_eval
# LANG: _005L --> _005N
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v223__005N = v222__005L.reshape((1, 2, 3))

# op _005O expand_array_eval
# LANG: frame_vel --> _005P
# SHAPES: (1, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v224__005P = np.einsum('ac,b->abc', v542_frame_vel.reshape((1, 3)) ,np.ones((2,))).reshape((1, 2, 3))

# op _006g expand_array_eval
# LANG: _006f --> _006h
# SHAPES: (1, 1, 2) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v241__006h = np.einsum('abc,d->abcd', v240__006f.reshape((1, 1, 2)) ,np.ones((3,))).reshape((1, 1, 2, 3))

# op _00bK_power_combination_eval
# LANG: _00bJ, _00aS --> _00bL
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v418__00bL = (v417__00bJ)*(v390__00aS)
v418__00bL = v418__00bL.reshape((1, 12, 3))

# op _00bO_linear_combination_eval
# LANG: _00bN, _00aO --> _00bP
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v420__00bP = v419__00bN+v388__00aO

# op _00f7_power_combination_eval
# LANG: _00f6, _00eN --> _00f8
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v529__00f8 = (v528__00f6)*(v518__00eN)
v529__00f8 = v529__00f8.reshape((1, 4, 3))

# op _00fb_linear_combination_eval
# LANG: _00fa, _00eJ --> _00fc
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v531__00fc = v530__00fa+v516__00eJ

# op _005R_linear_combination_eval
# LANG: _005N, _005P --> _005S
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v226__005S = v223__005N+v224__005P

# op _005T reshape_eval
# LANG: wing_coll_vel --> _005U
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v227__005U = v225_wing_coll_vel.reshape((1, 2, 3))

# op _006i_power_combination_eval
# LANG: _0069, _006h --> wing_bd_vtx_normals
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v535_wing_bd_vtx_normals = (v237__0069)*(v241__006h**-1)
v535_wing_bd_vtx_normals = v535_wing_bd_vtx_normals.reshape((1, 1, 2, 3))

# op _00bQ_linear_combination_eval
# LANG: _00bP, _00bL --> aic_M00
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v421_aic_M00 = v420__00bP+v418__00bL

# op _00fd_linear_combination_eval
# LANG: _00fc, _00f8 --> aic_bd00
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v532_aic_bd00 = v531__00fc+v529__00f8

# op _005V_linear_combination_eval
# LANG: _005S, _005U --> _005W
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v228__005W = v226__005S+v227__005U

# op _006x reshape_eval
# LANG: aic_M00 --> _006y
# SHAPES: (1, 12, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v252__006y = v421_aic_M00.reshape((1, 2, 6, 3))

# op _00bV reshape_eval
# LANG: wing_bd_vtx_normals --> _00bW
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v425__00bW = v535_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _00c4 reshape_eval
# LANG: aic_bd00 --> _00c5
# SHAPES: (1, 4, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v431__00c5 = v532_aic_bd00.reshape((1, 2, 2, 3))

# op _00fi reshape_eval
# LANG: wing_bd_vtx_normals --> _00fj
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v536__00fj = v535_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _005X_linear_combination_eval
# LANG: _005W --> wing_kinematic_vel
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v244_wing_kinematic_vel = -1*v228__005W

# op _006n reshape_eval
# LANG: wing_bd_vtx_normals --> _006o
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v246__006o = v535_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _006z_indexed_passthrough_eval
# LANG: _006y --> aic_M
# SHAPES: (1, 2, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v423_aic_M__temp[i_v252__006y__006z_indexed_passthrough_eval] = v252__006y.flatten()
v423_aic_M = v423_aic_M__temp.copy()

# op _00bX_indexed_passthrough_eval
# LANG: _00bW --> normal_concatenated_M_mat
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v422_normal_concatenated_M_mat__temp[i_v425__00bW__00bX_indexed_passthrough_eval] = v425__00bW.flatten()
v422_normal_concatenated_M_mat = v422_normal_concatenated_M_mat__temp.copy()

# op _00c6_indexed_passthrough_eval
# LANG: _00c5 --> aic_bd
# SHAPES: (1, 2, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v534_aic_bd__temp[i_v431__00c5__00c6_indexed_passthrough_eval] = v431__00c5.flatten()
v534_aic_bd = v534_aic_bd__temp.copy()

# op _00fk_indexed_passthrough_eval
# LANG: _00fj --> normal_concatenated_aic_bd_proj
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v533_normal_concatenated_aic_bd_proj__temp[i_v536__00fj__00fk_indexed_passthrough_eval] = v536__00fj.flatten()
v533_normal_concatenated_aic_bd_proj = v533_normal_concatenated_aic_bd_proj__temp.copy()

# op _006p_custom_explicit_eval
# LANG: _006o, wing_kinematic_vel --> b
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
temp = _006p_custom_explicit_func_b.solve(v244_wing_kinematic_vel, v246__006o)
v247_b = temp[0].copy()

# op _00bY_custom_explicit_eval
# LANG: normal_concatenated_M_mat, aic_M --> M_mat
# SHAPES: (1, 2, 3), (1, 2, 6, 3) --> (1, 2, 6)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
temp = _00bY_custom_explicit_func_M_mat.solve(v423_aic_M, v422_normal_concatenated_M_mat)
v426_M_mat = temp[0].copy()

# op _00fl_custom_explicit_eval
# LANG: normal_concatenated_aic_bd_proj, aic_bd --> aic_bd_proj
# SHAPES: (1, 2, 3), (1, 2, 2, 3) --> (1, 2, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
temp = _00fl_custom_explicit_func_aic_bd_proj.solve(v534_aic_bd, v533_normal_concatenated_aic_bd_proj)
v537_aic_bd_proj = temp[0].copy()

# op _004R_indexed_passthrough_eval
# LANG: wing_gamma_w --> gamma_w
# SHAPES: (1, 3, 2) --> (1, 3, 2)
# full namespace: combine_gamma_w
v200_gamma_w__temp[i_v188_wing_gamma_w__004R_indexed_passthrough_eval] = v188_wing_gamma_w.flatten()
v200_gamma_w = v200_gamma_w__temp.copy()

# op _005c_newton_implict_eval
# LANG: M_mat, b, aic_bd_proj, gamma_w --> gamma_b
# SHAPES: (1, 2, 6), (1, 2), (1, 2, 2), (1, 3, 2) --> (1, 2)
# full namespace: solve_gamma_b_group
_005c_newton.set_guess(initial_guess_v538_gamma_b)
_005c_newton_out = _005c_newton.solve(v537_aic_bd_proj, v426_M_mat, v200_gamma_w, v247_b)
v538_gamma_b = _005c_newton_out[0]

# op _00fv_linear_combination_eval
# LANG: frame_vel --> _00fw
# SHAPES: (1, 3) --> (1, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v543__00fw = -1*v542_frame_vel

# op _00fx expand_array_eval
# LANG: _00fw --> wing_wake_kinematic_vel
# SHAPES: (1, 3) --> (1, 3, 3, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v544_wing_wake_kinematic_vel = np.einsum('ad,bc->abcd', v543__00fw.reshape((1, 3)) ,np.ones((3, 3))).reshape((1, 3, 3, 3))

# op _00fs_linear_combination_eval
# LANG: wing_wake_kinematic_vel --> wing_wake_total_vel
# SHAPES: (1, 3, 3, 3) --> (1, 3, 3, 3)
# full namespace: ComputeWakeTotalVel
v541_wing_wake_total_vel = v544_wing_wake_kinematic_vel

# op _0016_decompose_eval
# LANG: wing_wake_total_vel --> _001P, _0017
# SHAPES: (1, 3, 3, 3) --> (1, 2, 3, 3), (1, 1, 3, 3)
# full namespace: 
v53__0017 = ((v541_wing_wake_total_vel.flatten())[src_indices__0017__0016]).reshape((1, 1, 3, 3))
v76__001P = ((v541_wing_wake_total_vel.flatten())[src_indices__001P__0016]).reshape((1, 2, 3, 3))

# op _001x_power_combination_eval
# LANG: _0017 --> _001y
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v66__001y = (v53__0017)
v66__001y = (v66__001y*_001x_coeff).reshape((1, 1, 3, 3))

# op _0014_decompose_eval
# LANG: wing_bd_vtx_coords --> _0015
# SHAPES: (1, 2, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v52__0015 = ((v433_wing_bd_vtx_coords.flatten())[src_indices__0015__0014]).reshape((1, 1, 3, 3))

# op _001z_power_combination_eval
# LANG: _001y --> _001A
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v67__001A = (v66__001y)
v67__001A = (v67__001A*_001z_coeff).reshape((1, 1, 3, 3))

# op _001B_linear_combination_eval
# LANG: _0015, _001A --> _001C
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v68__001C = v52__0015+v67__001A

# op _001e_decompose_eval
# LANG: wing_wake_coords --> _001M, _001f, _001J
# SHAPES: (1, 3, 3, 3) --> (1, 2, 3, 3), (1, 1, 3, 3), (1, 2, 3, 3)
# full namespace: 
v57__001f = ((v206_wing_wake_coords.flatten())[src_indices__001f__001e]).reshape((1, 1, 3, 3))
v72__001J = ((v206_wing_wake_coords.flatten())[src_indices__001J__001e]).reshape((1, 2, 3, 3))
v74__001M = ((v206_wing_wake_coords.flatten())[src_indices__001M__001e]).reshape((1, 2, 3, 3))

# op _003i_linear_combination_eval
# LANG: _002D, _002G --> _003j
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v133__003j = v108__002D+-1*v110__002G

# op _001D_linear_combination_eval
# LANG: _001C, _001f --> _001E
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v69__001E = v68__001C+-1*v57__001f

# op _0025_power_combination_eval
# LANG: u --> _0026
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v90__0026 = (v80_u**2)
v90__0026 = v90__0026.reshape((1, 1))

# op _0027_power_combination_eval
# LANG: v --> _0028
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v91__0028 = (v81_v**2)
v91__0028 = v91__0028.reshape((1, 1))

# op _002z_linear_combination_eval
# LANG: wing --> _002A
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: MeshPreprocessing_comp
v106__002A = v230_wing

# op _003k pnorm_axis_eval
# LANG: _003j --> _003l
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3)
# full namespace: MeshPreprocessing_comp
v134__003l = np.sum(v133__003j**2,axis=(3,))**(1 / 2)

# op _001F reshape_eval
# LANG: _001E --> _001G
# SHAPES: (1, 1, 3, 3) --> (1, 3, 3)
# full namespace: 
v70__001G = v69__001E.reshape((1, 3, 3))

# op _0029_linear_combination_eval
# LANG: _0026, _0028 --> _002a
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v92__002a = v90__0026+v91__0028

# op _002b_power_combination_eval
# LANG: w --> _002c
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v93__002c = (v114_w**2)
v93__002c = v93__002c.reshape((1, 1))

# op _003G_decompose_eval
# LANG: _002A --> _003M, _003H, _003I, _003L
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v147__003H = ((v106__002A.flatten())[src_indices__003H__003G]).reshape((1, 1, 2, 3))
v148__003I = ((v106__002A.flatten())[src_indices__003I__003G]).reshape((1, 1, 2, 3))
v150__003L = ((v106__002A.flatten())[src_indices__003L__003G]).reshape((1, 1, 2, 3))
v151__003M = ((v106__002A.flatten())[src_indices__003M__003G]).reshape((1, 1, 2, 3))

# op _003m_decompose_eval
# LANG: _003l --> _003o, _003n
# SHAPES: (1, 1, 3) --> (1, 1, 2), (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v135__003n = ((v134__003l.flatten())[src_indices__003n__003m]).reshape((1, 1, 2))
v136__003o = ((v134__003l.flatten())[src_indices__003o__003m]).reshape((1, 1, 2))

# op _0040_power_combination_eval
# LANG: _003_ --> _0041
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v160__0041 = (v159__003_)
v160__0041 = (v160__0041*_0040_coeff).reshape((1, 1, 2, 3))

# op _0043_power_combination_eval
# LANG: _0042 --> _0044
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v162__0044 = (v161__0042)
v162__0044 = (v162__0044*_0043_coeff).reshape((1, 1, 2, 3))

# op _0018_power_combination_eval
# LANG: _0017 --> _0019
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v54__0019 = (v53__0017)
v54__0019 = (v54__0019*_0018_coeff).reshape((1, 1, 3, 3))

# op _001H expand_array_eval
# LANG: _001G --> _001I
# SHAPES: (1, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v71__001I = np.einsum('acd,b->abcd', v70__001G.reshape((1, 3, 3)) ,np.ones((2,))).reshape((1, 2, 3, 3))

# op _002d_linear_combination_eval
# LANG: _002a, _002c --> v_inf_sq
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v179_v_inf_sq = v92__002a+v93__002c

# op _003J_linear_combination_eval
# LANG: _003H, _003I --> _003K
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v149__003K = v147__003H+-1*v148__003I

# op _003N_linear_combination_eval
# LANG: _003L, _003M --> _003O
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v152__003O = v150__003L+-1*v151__003M

# op _003p_linear_combination_eval
# LANG: _003n, _003o --> _003q
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v137__003q = v135__003n+v136__003o

# op _0045_linear_combination_eval
# LANG: _0041, _0044 --> _0046
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v163__0046 = v160__0041+v162__0044

# op _0048_power_combination_eval
# LANG: _0047 --> _0049
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v165__0049 = (v164__0047)
v165__0049 = (v165__0049*_0048_coeff).reshape((1, 1, 2, 3))

# op _00fo_decompose_eval
# LANG: gamma_b --> wing_gamma_b
# SHAPES: (1, 2) --> (1, 2)
# full namespace: seperate_gamma_b
v539_wing_gamma_b = ((v538_gamma_b.flatten())[src_indices_wing_gamma_b__00fo]).reshape((1, 2))

# op _000I_decompose_eval
# LANG: wing_gamma_b --> _000J
# SHAPES: (1, 2) --> (1, 2)
# full namespace: 
v38__000J = ((v539_wing_gamma_b.flatten())[src_indices__000J__000I]).reshape((1, 2))

# op _001K_linear_combination_eval
# LANG: _001I, _001J --> _001L
# SHAPES: (1, 2, 3, 3), (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v73__001L = v71__001I+v72__001J

# op _001a_power_combination_eval
# LANG: _0019 --> _001b
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v55__001b = (v54__0019)
v55__001b = (v55__001b*_001a_coeff).reshape((1, 1, 3, 3))

# op _003P cross_product_eval
# LANG: _003K, _003O --> _003Q
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v153__003Q = np.cross(v149__003K, v152__003O, axisa = 3, axisb = 3, axisc = 3)

# op _003r_power_combination_eval
# LANG: _003q --> wing_chord_length
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v138_wing_chord_length = (v137__003q)
v138_wing_chord_length = (v138_wing_chord_length*_003r_coeff).reshape((1, 1, 2))

# op _003v_linear_combination_eval
# LANG: _003t, _003u --> _003w
# SHAPES: (1, 2, 2, 3), (1, 2, 2, 3) --> (1, 2, 2, 3)
# full namespace: MeshPreprocessing_comp
v141__003w = v139__003t+-1*v140__003u

# op _004F_power_combination_eval
# LANG: v_inf_sq --> _004G
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v182__004G = (v179_v_inf_sq**0.5)
v182__004G = v182__004G.reshape((1, 1))

# op _004a_linear_combination_eval
# LANG: _0046, _0049 --> _004b
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v166__004b = v163__0046+v165__0049

# op _004c_power_combination_eval
# LANG: _003b --> _004d
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v167__004d = (v129__003b)
v167__004d = (v167__004d*_004c_coeff).reshape((1, 1, 2, 3))

# op _000K reshape_eval
# LANG: _000J --> _000L
# SHAPES: (1, 2) --> (1, 1, 2)
# full namespace: 
v39__000L = v38__000J.reshape((1, 1, 2))

# op _000M_decompose_eval
# LANG: wing_gamma_w --> _000U, _000N, _000T
# SHAPES: (1, 3, 2) --> (1, 2, 2), (1, 1, 2), (1, 2, 2)
# full namespace: 
v40__000N = ((v188_wing_gamma_w.flatten())[src_indices__000N__000M]).reshape((1, 1, 2))
v43__000T = ((v188_wing_gamma_w.flatten())[src_indices__000T__000M]).reshape((1, 2, 2))
v44__000U = ((v188_wing_gamma_w.flatten())[src_indices__000U__000M]).reshape((1, 2, 2))

# op _001N_linear_combination_eval
# LANG: _001L, _001M --> _001O
# SHAPES: (1, 2, 3, 3), (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v75__001O = v73__001L+-1*v74__001M

# op _001Q_power_combination_eval
# LANG: _001P --> _001R
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v77__001R = (v76__001P)
v77__001R = (v77__001R*_001Q_coeff).reshape((1, 2, 3, 3))

# op _001c_linear_combination_eval
# LANG: _0015, _001b --> _001d
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v56__001d = v52__0015+v55__001b

# op _003R_power_combination_eval
# LANG: _003Q --> _003S
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v154__003S = (v153__003Q**2)
v154__003S = v154__003S.reshape((1, 1, 2, 3))

# op _003x pnorm_axis_eval
# LANG: _003w --> _003y
# SHAPES: (1, 2, 2, 3) --> (1, 2, 2)
# full namespace: MeshPreprocessing_comp
v142__003y = np.sum(v141__003w**2,axis=(3,))**(1 / 2)

# op _004B_single_tensor_sum_with_axis_eval
# LANG: wing_chord_length --> _004C
# SHAPES: (1, 1, 2) --> (1, 2)
# full namespace: MeshPreprocessing_comp
v180__004C = np.sum(v138_wing_chord_length, axis = (1,)).reshape((1, 2))

# op _004H_power_combination_eval
# LANG: density, _004G --> _004I
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v183__004I = (v178_density)*(v182__004G)
v183__004I = v183__004I.reshape((1, 1))

# op _004e_linear_combination_eval
# LANG: _004b, _004d --> _004f
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v168__004f = v166__004b+v167__004d

# op _004l_power_combination_eval
# LANG: _003_ --> _004m
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v171__004m = (v159__003_)
v171__004m = (v171__004m*_004l_coeff).reshape((1, 1, 2, 3))

# op _004n_power_combination_eval
# LANG: _0047 --> _004o
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v172__004o = (v164__0047)
v172__004o = (v172__004o*_004n_coeff).reshape((1, 1, 2, 3))

# op _000O_linear_combination_eval
# LANG: _000L, _000N --> _000P
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: 
v41__000P = v39__000L+-1*v40__000N

# op _000V_linear_combination_eval
# LANG: _000T, _000U --> _000W
# SHAPES: (1, 2, 2), (1, 2, 2) --> (1, 2, 2)
# full namespace: 
v45__000W = v43__000T+-1*v44__000U

# op _001S_linear_combination_eval
# LANG: _001O, _001R --> _001T
# SHAPES: (1, 2, 3, 3), (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v78__001T = v75__001O+v77__001R

# op _001g_linear_combination_eval
# LANG: _001d, _001f --> _001h
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v58__001h = v56__001d+-1*v57__001f

# op _003T_single_tensor_sum_with_axis_eval
# LANG: _003S --> _003U
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v155__003U = np.sum(v154__003S, axis = (3,)).reshape((1, 1, 2))

# op _003z_decompose_eval
# LANG: _003y --> _003B, _003A
# SHAPES: (1, 2, 2) --> (1, 1, 2), (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v143__003A = ((v142__003y.flatten())[src_indices__003A__003z]).reshape((1, 1, 2))
v144__003B = ((v142__003y.flatten())[src_indices__003B__003z]).reshape((1, 1, 2))

# op _004D reshape_eval
# LANG: _004C --> _004E
# SHAPES: (1, 2) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v181__004E = v180__004C.reshape((1, 2, 1))

# op _004J expand_array_eval
# LANG: _004I --> _004K
# SHAPES: (1, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v184__004K = np.einsum('ac,b->abc', v183__004I.reshape((1, 1)) ,np.ones((2,))).reshape((1, 2, 1))

# op _004g reshape_eval
# LANG: _004f --> _004h
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v169__004h = v168__004f.reshape((1, 2, 3))

# op _004p_linear_combination_eval
# LANG: _004m, _004o --> _004q
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v173__004q = v171__004m+v172__004o

# op _004r_power_combination_eval
# LANG: _0042 --> _004s
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v174__004s = (v161__0042)
v174__004s = (v174__004s*_004r_coeff).reshape((1, 1, 2, 3))

# op _000Q_power_combination_eval
# LANG: _000P --> _000R
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: 
v42__000R = (v41__000P)
v42__000R = (v42__000R*_000Q_coeff).reshape((1, 1, 2))

# op _000X_power_combination_eval
# LANG: _000W --> _000Y
# SHAPES: (1, 2, 2) --> (1, 2, 2)
# full namespace: 
v46__000Y = (v45__000W)
v46__000Y = (v46__000Y*_000X_coeff).reshape((1, 2, 2))

# op _001U_power_combination_eval
# LANG: _001T --> _001V
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v79__001V = (v78__001T)
v79__001V = (v79__001V*_001U_coeff).reshape((1, 2, 3, 3))

# op _001i_power_combination_eval
# LANG: _001h --> _001j
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v59__001j = (v58__001h)
v59__001j = (v59__001j*_001i_coeff).reshape((1, 1, 3, 3))

# op _003C_linear_combination_eval
# LANG: _003A, _003B --> _003D
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v145__003D = v143__003A+v144__003B

# op _003V_power_combination_eval
# LANG: _003U --> _003W
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v156__003W = (v155__003U**0.5)
v156__003W = v156__003W.reshape((1, 1, 2))

# op _004L_power_combination_eval
# LANG: _004K, _004E --> _004M
# SHAPES: (1, 2, 1), (1, 2, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v185__004M = (v184__004K)*(v181__004E)
v185__004M = v185__004M.reshape((1, 2, 1))

# op _004i_linear_combination_eval
# LANG: _004h --> _004j
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v170__004j = -1*v169__004h

# op _004t_linear_combination_eval
# LANG: _004q, _004s --> _004u
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v175__004u = v173__004q+v174__004s

# op _004v_power_combination_eval
# LANG: _003b --> _004w
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v176__004w = (v129__003b)
v176__004w = (v176__004w*_004v_coeff).reshape((1, 1, 2, 3))

# op _000S_indexed_passthrough_eval
# LANG: _000R, _000Y --> wing_dgammaw_dt
# SHAPES: (1, 1, 2), (1, 2, 2) --> (1, 3, 2)
# full namespace: 
v37_wing_dgammaw_dt__temp[i_v42__000R__000S_indexed_passthrough_eval] = v42__000R.flatten()
v37_wing_dgammaw_dt = v37_wing_dgammaw_dt__temp.copy()
v37_wing_dgammaw_dt__temp[i_v46__000Y__000S_indexed_passthrough_eval] = v46__000Y.flatten()
v37_wing_dgammaw_dt = v37_wing_dgammaw_dt__temp.copy()

# op _001k_indexed_passthrough_eval
# LANG: _001j, _001V --> wing_dwake_coords_dt
# SHAPES: (1, 1, 3, 3), (1, 2, 3, 3) --> (1, 3, 3, 3)
# full namespace: 
v51_wing_dwake_coords_dt__temp[i_v59__001j__001k_indexed_passthrough_eval] = v59__001j.flatten()
v51_wing_dwake_coords_dt = v51_wing_dwake_coords_dt__temp.copy()
v51_wing_dwake_coords_dt__temp[i_v79__001V__001k_indexed_passthrough_eval] = v79__001V.flatten()
v51_wing_dwake_coords_dt = v51_wing_dwake_coords_dt__temp.copy()

# op _002r_linear_combination_eval
# LANG: theta, gamma --> alpha
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v101_alpha = v86_theta+-1*v88_gamma

# op _002t_linear_combination_eval
# LANG: psi, psiw --> beta
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v102_beta = v87_psi+v89_psiw

# op _003E_power_combination_eval
# LANG: _003D --> wing_span_length
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v146_wing_span_length = (v145__003D)
v146_wing_span_length = (v146_wing_span_length*_003E_coeff).reshape((1, 1, 2))

# op _003X_power_combination_eval
# LANG: _003W --> wing_s_panel
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v157_wing_s_panel = (v156__003W)
v157_wing_s_panel = (v157_wing_s_panel*_003X_coeff).reshape((1, 1, 2))

# op _004N_power_combination_eval
# LANG: _004M --> wing_re_span
# SHAPES: (1, 2, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v186_wing_re_span = (v185__004M)
v186_wing_re_span = (v186_wing_re_span*_004N_coeff).reshape((1, 2, 1))

# op _004k_indexed_passthrough_eval
# LANG: _004j --> bd_vec
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v158_bd_vec__temp[i_v170__004j__004k_indexed_passthrough_eval] = v170__004j.flatten()
v158_bd_vec = v158_bd_vec__temp.copy()

# op _004x_linear_combination_eval
# LANG: _004u, _004w --> wing_eval_pts_coords
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v177_wing_eval_pts_coords = v175__004u+v176__004w

# op _006r_indexed_passthrough_eval
# LANG: _006o --> normal_concatenated_b
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v243_normal_concatenated_b__temp[i_v246__006o__006r_indexed_passthrough_eval] = v246__006o.flatten()
v243_normal_concatenated_b = v243_normal_concatenated_b__temp.copy()