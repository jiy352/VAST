

# RUN_MODEL_ode_system

# system evaluation block

# op _002w_linear_combination_eval
# LANG: u --> _002x
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v104__002x = -1*v80_u

# op _002z_linear_combination_eval
# LANG: w --> _002A
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v105__002A = -1*v116_w

# op _002y_indexed_passthrough_eval
# LANG: _002x, _002A --> frame_vel
# SHAPES: (1, 1), (1, 1) --> (1, 3)
# full namespace: adapter_comp
v544_frame_vel__temp[i_v104__002x__002y_indexed_passthrough_eval] = v104__002x.flatten()
v544_frame_vel = v544_frame_vel__temp.copy()
v544_frame_vel__temp[i_v105__002A__002y_indexed_passthrough_eval] = v105__002A.flatten()
v544_frame_vel = v544_frame_vel__temp.copy()

# op _002T_decompose_eval
# LANG: frame_vel --> _002Y, _002U
# SHAPES: (1, 3) --> (1, 1), (1, 1)
# full namespace: MeshPreprocessing_comp
v118__002U = ((v544_frame_vel.flatten())[src_indices__002U__002T]).reshape((1, 1))
v120__002Y = ((v544_frame_vel.flatten())[src_indices__002Y__002T]).reshape((1, 1))

# op _002V_linear_combination_eval
# LANG: _002U --> _002W
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v119__002W = -1*v118__002U

# op _002Z_linear_combination_eval
# LANG: _002Y --> _002_
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v121__002_ = -1*v120__002Y

# op _002G_decompose_eval
# LANG: wing --> _003f, _002H, _002K, _0038, _0039, _003e, _003x, _003y, _0043, _0046, _004b
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 2, 2, 3), (1, 2, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v110__002H = ((v232_wing.flatten())[src_indices__002H__002G]).reshape((1, 1, 3, 3))
v112__002K = ((v232_wing.flatten())[src_indices__002K__002G]).reshape((1, 1, 3, 3))
v126__0038 = ((v232_wing.flatten())[src_indices__0038__002G]).reshape((1, 1, 2, 3))
v127__0039 = ((v232_wing.flatten())[src_indices__0039__002G]).reshape((1, 1, 2, 3))
v130__003e = ((v232_wing.flatten())[src_indices__003e__002G]).reshape((1, 1, 2, 3))
v131__003f = ((v232_wing.flatten())[src_indices__003f__002G]).reshape((1, 1, 2, 3))
v141__003x = ((v232_wing.flatten())[src_indices__003x__002G]).reshape((1, 2, 2, 3))
v142__003y = ((v232_wing.flatten())[src_indices__003y__002G]).reshape((1, 2, 2, 3))
v161__0043 = ((v232_wing.flatten())[src_indices__0043__002G]).reshape((1, 1, 2, 3))
v163__0046 = ((v232_wing.flatten())[src_indices__0046__002G]).reshape((1, 1, 2, 3))
v166__004b = ((v232_wing.flatten())[src_indices__004b__002G]).reshape((1, 1, 2, 3))

# op _002X_indexed_passthrough_eval
# LANG: _002W, _002_, w --> fs
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v117_fs__temp[i_v119__002W__002X_indexed_passthrough_eval] = v119__002W.flatten()
v117_fs = v117_fs__temp.copy()
v117_fs__temp[i_v121__002___002X_indexed_passthrough_eval] = v121__002_.flatten()
v117_fs = v117_fs__temp.copy()
v117_fs__temp[i_v116_w__002X_indexed_passthrough_eval] = v116_w.flatten()
v117_fs = v117_fs__temp.copy()

# op _005q_decompose_eval
# LANG: wing_wake_coords --> _005r
# SHAPES: (1, 3, 3, 3) --> (1, 1, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v210__005r = ((v208_wing_wake_coords.flatten())[src_indices__005r__005q]).reshape((1, 1, 3, 3))

# op _0030_power_combination_eval
# LANG: fs --> _0031
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v122__0031 = (v117_fs)
v122__0031 = (v122__0031*_0030_coeff).reshape((1, 3))

# op _003a_linear_combination_eval
# LANG: _0038, _0039 --> _003b
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v128__003b = v126__0038+v127__0039

# op _003g_linear_combination_eval
# LANG: _003f, _003e --> _003h
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v132__003h = v130__003e+v131__003f

# op _005s_power_combination_eval
# LANG: _005r --> _005t
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v211__005t = (v210__005r)
v211__005t = v211__005t.reshape((1, 1, 3, 3))

# op _0032_power_combination_eval
# LANG: _0031 --> _0033
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v123__0033 = (v122__0031)
v123__0033 = (v123__0033*_0032_coeff).reshape((1, 3))

# op _003c_power_combination_eval
# LANG: _003b --> _003d
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v129__003d = (v128__003b)
v129__003d = (v129__003d*_003c_coeff).reshape((1, 1, 2, 3))

# op _003i_power_combination_eval
# LANG: _003h --> _003j
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v133__003j = (v132__003h)
v133__003j = (v133__003j*_003i_coeff).reshape((1, 1, 2, 3))

# op _005p_indexed_passthrough_eval
# LANG: _005t, wing_wake_coords --> wing_TE_wake_coords
# SHAPES: (1, 1, 3, 3), (1, 3, 3, 3) --> (1, 4, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v256_wing_TE_wake_coords__temp[i_v208_wing_wake_coords__005p_indexed_passthrough_eval] = v208_wing_wake_coords.flatten()
v256_wing_TE_wake_coords = v256_wing_TE_wake_coords__temp.copy()
v256_wing_TE_wake_coords__temp[i_v211__005t__005p_indexed_passthrough_eval] = v211__005t.flatten()
v256_wing_TE_wake_coords = v256_wing_TE_wake_coords__temp.copy()

# op _002I_power_combination_eval
# LANG: _002H --> _002J
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v111__002J = (v110__002H)
v111__002J = (v111__002J*_002I_coeff).reshape((1, 1, 3, 3))

# op _002L_power_combination_eval
# LANG: _002K --> _002M
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v113__002M = (v112__002K)
v113__002M = (v113__002M*_002L_coeff).reshape((1, 1, 3, 3))

# op _0034 expand_array_eval
# LANG: _0033 --> _0035
# SHAPES: (1, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v124__0035 = np.einsum('ad,bc->abcd', v123__0033.reshape((1, 3)) ,np.ones((1, 3))).reshape((1, 1, 3, 3))

# op _003k_linear_combination_eval
# LANG: _003d, _003j --> wing_coll_pts_coords
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v434_wing_coll_pts_coords = v129__003d+v133__003j

# op _006G_decompose_eval
# LANG: wing_TE_wake_coords --> _006H, _006I, _006J, _006K
# SHAPES: (1, 4, 3, 3) --> (1, 3, 2, 3), (1, 3, 2, 3), (1, 3, 2, 3), (1, 3, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v257__006H = ((v256_wing_TE_wake_coords.flatten())[src_indices__006H__006G]).reshape((1, 3, 2, 3))
v258__006I = ((v256_wing_TE_wake_coords.flatten())[src_indices__006I__006G]).reshape((1, 3, 2, 3))
v259__006J = ((v256_wing_TE_wake_coords.flatten())[src_indices__006J__006G]).reshape((1, 3, 2, 3))
v260__006K = ((v256_wing_TE_wake_coords.flatten())[src_indices__006K__006G]).reshape((1, 3, 2, 3))

# op _002N_linear_combination_eval
# LANG: _002J, _002M --> _002O
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v114__002O = v111__002J+v113__002M

# op _0036_linear_combination_eval
# LANG: _002K, _0035 --> _0037
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v125__0037 = v112__002K+v124__0035

# op _006L reshape_eval
# LANG: wing_coll_pts_coords --> _006M
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v261__006M = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _006R reshape_eval
# LANG: _006H --> _006S
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v264__006S = v257__006H.reshape((1, 6, 3))

# op _0074 reshape_eval
# LANG: wing_coll_pts_coords --> _0075
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v271__0075 = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007a reshape_eval
# LANG: _006I --> _007b
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v274__007b = v258__006I.reshape((1, 6, 3))

# op _007o reshape_eval
# LANG: wing_coll_pts_coords --> _007p
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v281__007p = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007u reshape_eval
# LANG: _006J --> _007v
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v284__007v = v259__006J.reshape((1, 6, 3))

# op _002P_indexed_passthrough_eval
# LANG: _002O, _0037 --> wing_bd_vtx_coords
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 2, 3, 3)
# full namespace: MeshPreprocessing_comp
v435_wing_bd_vtx_coords__temp[i_v114__002O__002P_indexed_passthrough_eval] = v114__002O.flatten()
v435_wing_bd_vtx_coords = v435_wing_bd_vtx_coords__temp.copy()
v435_wing_bd_vtx_coords__temp[i_v125__0037__002P_indexed_passthrough_eval] = v125__0037.flatten()
v435_wing_bd_vtx_coords = v435_wing_bd_vtx_coords__temp.copy()

# op _006N expand_array_eval
# LANG: _006M --> _006O
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v262__006O = np.einsum('abd,c->abcd', v261__006M.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _006T expand_array_eval
# LANG: _006S --> _006U
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v265__006U = np.einsum('acd,b->abcd', v264__006S.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _0076 expand_array_eval
# LANG: _0075 --> _0077
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v272__0077 = np.einsum('abd,c->abcd', v271__0075.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007I reshape_eval
# LANG: wing_coll_pts_coords --> _007J
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v291__007J = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007O reshape_eval
# LANG: _006K --> _007P
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v294__007P = v260__006K.reshape((1, 6, 3))

# op _007c expand_array_eval
# LANG: _007b --> _007d
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v275__007d = np.einsum('acd,b->abcd', v274__007b.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _007q expand_array_eval
# LANG: _007p --> _007r
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v282__007r = np.einsum('abd,c->abcd', v281__007p.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007w expand_array_eval
# LANG: _007v --> _007x
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v285__007x = np.einsum('acd,b->abcd', v284__007v.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _006P reshape_eval
# LANG: _006O --> _006Q
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v263__006Q = v262__006O.reshape((1, 12, 3))

# op _006V reshape_eval
# LANG: _006U --> _006W
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v266__006W = v265__006U.reshape((1, 12, 3))

# op _0078 reshape_eval
# LANG: _0077 --> _0079
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v273__0079 = v272__0077.reshape((1, 12, 3))

# op _007K expand_array_eval
# LANG: _007J --> _007L
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v292__007L = np.einsum('abd,c->abcd', v291__007J.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007Q expand_array_eval
# LANG: _007P --> _007R
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v295__007R = np.einsum('acd,b->abcd', v294__007P.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _007e reshape_eval
# LANG: _007d --> _007f
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v276__007f = v275__007d.reshape((1, 12, 3))

# op _007s reshape_eval
# LANG: _007r --> _007t
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v283__007t = v282__007r.reshape((1, 12, 3))

# op _007y reshape_eval
# LANG: _007x --> _007z
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v286__007z = v285__007x.reshape((1, 12, 3))

# op _00cd_decompose_eval
# LANG: wing_bd_vtx_coords --> _00ce, _00cf, _00cg, _00ch
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v436__00ce = ((v435_wing_bd_vtx_coords.flatten())[src_indices__00ce__00cd]).reshape((1, 1, 2, 3))
v437__00cf = ((v435_wing_bd_vtx_coords.flatten())[src_indices__00cf__00cd]).reshape((1, 1, 2, 3))
v438__00cg = ((v435_wing_bd_vtx_coords.flatten())[src_indices__00cg__00cd]).reshape((1, 1, 2, 3))
v439__00ch = ((v435_wing_bd_vtx_coords.flatten())[src_indices__00ch__00cd]).reshape((1, 1, 2, 3))

# op _006X_linear_combination_eval
# LANG: _006Q, _006W --> _006Y
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v267__006Y = v263__006Q+-1*v266__006W

# op _007A_linear_combination_eval
# LANG: _007t, _007z --> _007B
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v287__007B = v283__007t+-1*v286__007z

# op _007M reshape_eval
# LANG: _007L --> _007N
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v293__007N = v292__007L.reshape((1, 12, 3))

# op _007S reshape_eval
# LANG: _007R --> _007T
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v296__007T = v295__007R.reshape((1, 12, 3))

# op _007g_linear_combination_eval
# LANG: _0079, _007f --> _007h
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v277__007h = v273__0079+-1*v276__007f

# op _00cC reshape_eval
# LANG: wing_coll_pts_coords --> _00cD
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v450__00cD = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00cI reshape_eval
# LANG: _00cf --> _00cJ
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v453__00cJ = v437__00cf.reshape((1, 2, 3))

# op _00cW reshape_eval
# LANG: wing_coll_pts_coords --> _00cX
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v460__00cX = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00ci reshape_eval
# LANG: wing_coll_pts_coords --> _00cj
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v440__00cj = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00co reshape_eval
# LANG: _00ce --> _00cp
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v443__00cp = v436__00ce.reshape((1, 2, 3))

# op _00d1 reshape_eval
# LANG: _00cg --> _00d2
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v463__00d2 = v438__00cg.reshape((1, 2, 3))

# op _006Z_power_combination_eval
# LANG: _006Y --> _006_
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v268__006_ = (v267__006Y**2)
v268__006_ = v268__006_.reshape((1, 12, 3))

# op _007C_power_combination_eval
# LANG: _007B --> _007D
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v288__007D = (v287__007B**2)
v288__007D = v288__007D.reshape((1, 12, 3))

# op _007U_linear_combination_eval
# LANG: _007N, _007T --> _007V
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v297__007V = v293__007N+-1*v296__007T

# op _007i_power_combination_eval
# LANG: _007h --> _007j
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v278__007j = (v277__007h**2)
v278__007j = v278__007j.reshape((1, 12, 3))

# op _00cE expand_array_eval
# LANG: _00cD --> _00cF
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v451__00cF = np.einsum('abd,c->abcd', v450__00cD.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cK expand_array_eval
# LANG: _00cJ --> _00cL
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v454__00cL = np.einsum('acd,b->abcd', v453__00cJ.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cY expand_array_eval
# LANG: _00cX --> _00cZ
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v461__00cZ = np.einsum('abd,c->abcd', v460__00cX.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00ck expand_array_eval
# LANG: _00cj --> _00cl
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v441__00cl = np.einsum('abd,c->abcd', v440__00cj.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cq expand_array_eval
# LANG: _00cp --> _00cr
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v444__00cr = np.einsum('acd,b->abcd', v443__00cp.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00d3 expand_array_eval
# LANG: _00d2 --> _00d4
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v464__00d4 = np.einsum('acd,b->abcd', v463__00d2.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00df reshape_eval
# LANG: wing_coll_pts_coords --> _00dg
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v470__00dg = v434_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00dl reshape_eval
# LANG: _00ch --> _00dm
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v473__00dm = v439__00ch.reshape((1, 2, 3))

# op _0070_single_tensor_sum_with_axis_eval
# LANG: _006_ --> _0071
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v269__0071 = np.sum(v268__006_, axis = (2,)).reshape((1, 12))

# op _007E_single_tensor_sum_with_axis_eval
# LANG: _007D --> _007F
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v289__007F = np.sum(v288__007D, axis = (2,)).reshape((1, 12))

# op _007W_power_combination_eval
# LANG: _007V --> _007X
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v298__007X = (v297__007V**2)
v298__007X = v298__007X.reshape((1, 12, 3))

# op _007k_single_tensor_sum_with_axis_eval
# LANG: _007j --> _007l
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v279__007l = np.sum(v278__007j, axis = (2,)).reshape((1, 12))

# op _00cG reshape_eval
# LANG: _00cF --> _00cH
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v452__00cH = v451__00cF.reshape((1, 4, 3))

# op _00cM reshape_eval
# LANG: _00cL --> _00cN
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v455__00cN = v454__00cL.reshape((1, 4, 3))

# op _00c_ reshape_eval
# LANG: _00cZ --> _00d0
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v462__00d0 = v461__00cZ.reshape((1, 4, 3))

# op _00cm reshape_eval
# LANG: _00cl --> _00cn
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v442__00cn = v441__00cl.reshape((1, 4, 3))

# op _00cs reshape_eval
# LANG: _00cr --> _00ct
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v445__00ct = v444__00cr.reshape((1, 4, 3))

# op _00d5 reshape_eval
# LANG: _00d4 --> _00d6
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v465__00d6 = v464__00d4.reshape((1, 4, 3))

# op _00dh expand_array_eval
# LANG: _00dg --> _00di
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v471__00di = np.einsum('abd,c->abcd', v470__00dg.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00dn expand_array_eval
# LANG: _00dm --> _00do
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v474__00do = np.einsum('acd,b->abcd', v473__00dm.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _0072_power_combination_eval
# LANG: _0071 --> _0073
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v270__0073 = (v269__0071**0.5)
v270__0073 = v270__0073.reshape((1, 12))

# op _007G_power_combination_eval
# LANG: _007F --> _007H
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v290__007H = (v289__007F**0.5)
v290__007H = v290__007H.reshape((1, 12))

# op _007Y_single_tensor_sum_with_axis_eval
# LANG: _007X --> _007Z
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v299__007Z = np.sum(v298__007X, axis = (2,)).reshape((1, 12))

# op _007m_power_combination_eval
# LANG: _007l --> _007n
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v280__007n = (v279__007l**0.5)
v280__007n = v280__007n.reshape((1, 12))

# op _00cO_linear_combination_eval
# LANG: _00cH, _00cN --> _00cP
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v456__00cP = v452__00cH+-1*v455__00cN

# op _00cu_linear_combination_eval
# LANG: _00cn, _00ct --> _00cv
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v446__00cv = v442__00cn+-1*v445__00ct

# op _00d7_linear_combination_eval
# LANG: _00d0, _00d6 --> _00d8
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v466__00d8 = v462__00d0+-1*v465__00d6

# op _00dj reshape_eval
# LANG: _00di --> _00dk
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v472__00dk = v471__00di.reshape((1, 4, 3))

# op _00dp reshape_eval
# LANG: _00do --> _00dq
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v475__00dq = v474__00do.reshape((1, 4, 3))

# op _007__power_combination_eval
# LANG: _007Z --> _0080
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v300__0080 = (v299__007Z**0.5)
v300__0080 = v300__0080.reshape((1, 12))

# op _0085_power_combination_eval
# LANG: _006Y, _007h --> _0086
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v303__0086 = (v267__006Y)*(v277__007h)
v303__0086 = v303__0086.reshape((1, 12, 3))

# op _0089_power_combination_eval
# LANG: _0073 --> _008a
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v305__008a = (v270__0073**2)
v305__008a = v305__008a.reshape((1, 12))

# op _008H_power_combination_eval
# LANG: _0073 --> _008I
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v322__008I = (v270__0073)
v322__008I = (v322__008I*_008H_coeff).reshape((1, 12))

# op _008b_power_combination_eval
# LANG: _007n --> _008c
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v306__008c = (v280__007n**2)
v306__008c = v306__008c.reshape((1, 12))

# op _0092_power_combination_eval
# LANG: _007B, _007h --> _0093
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v333__0093 = (v277__007h)*(v287__007B)
v333__0093 = v333__0093.reshape((1, 12, 3))

# op _0096_power_combination_eval
# LANG: _007n --> _0097
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v335__0097 = (v280__007n**2)
v335__0097 = v335__0097.reshape((1, 12))

# op _0098_power_combination_eval
# LANG: _007H --> _0099
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v336__0099 = (v290__007H**2)
v336__0099 = v336__0099.reshape((1, 12))

# op _009E_power_combination_eval
# LANG: _007n --> _009F
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v352__009F = (v280__007n)
v352__009F = (v352__009F*_009E_coeff).reshape((1, 12))

# op _00cQ_power_combination_eval
# LANG: _00cP --> _00cR
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v457__00cR = (v456__00cP**2)
v457__00cR = v457__00cR.reshape((1, 4, 3))

# op _00cw_power_combination_eval
# LANG: _00cv --> _00cx
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v447__00cx = (v446__00cv**2)
v447__00cx = v447__00cx.reshape((1, 4, 3))

# op _00d9_power_combination_eval
# LANG: _00d8 --> _00da
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v467__00da = (v466__00d8**2)
v467__00da = v467__00da.reshape((1, 4, 3))

# op _00dr_linear_combination_eval
# LANG: _00dk, _00dq --> _00ds
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v476__00ds = v472__00dk+-1*v475__00dq

# op _0087_single_tensor_sum_with_axis_eval
# LANG: _0086 --> _0088
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v304__0088 = np.sum(v303__0086, axis = (2,)).reshape((1, 12))

# op _008F_linear_combination_eval
# LANG: _008a, _008c --> _008G
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v321__008G = v305__008a+v306__008c

# op _008J_power_combination_eval
# LANG: _007n, _008I --> _008K
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v323__008K = (v322__008I)*(v280__007n)
v323__008K = v323__008K.reshape((1, 12))

# op _008f_linear_combination_eval
# LANG: _008a --> _008g
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v308__008g = _008f_constant+v305__008a

# op _008p_linear_combination_eval
# LANG: _008c --> _008q
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v313__008q = _008p_constant+v306__008c

# op _0094_single_tensor_sum_with_axis_eval
# LANG: _0093 --> _0095
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v334__0095 = np.sum(v333__0093, axis = (2,)).reshape((1, 12))

# op _009C_linear_combination_eval
# LANG: _0097, _0099 --> _009D
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v351__009D = v335__0097+v336__0099

# op _009G_power_combination_eval
# LANG: _007H, _009F --> _009H
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v353__009H = (v352__009F)*(v290__007H)
v353__009H = v353__009H.reshape((1, 12))

# op _009__power_combination_eval
# LANG: _007V, _007B --> _00a0
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v363__00a0 = (v287__007B)*(v297__007V)
v363__00a0 = v363__00a0.reshape((1, 12, 3))

# op _009c_linear_combination_eval
# LANG: _0097 --> _009d
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v338__009d = _009c_constant+v335__0097

# op _009m_linear_combination_eval
# LANG: _0099 --> _009n
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v343__009n = _009m_constant+v336__0099

# op _00a3_power_combination_eval
# LANG: _007H --> _00a4
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v365__00a4 = (v290__007H**2)
v365__00a4 = v365__00a4.reshape((1, 12))

# op _00a5_power_combination_eval
# LANG: _0080 --> _00a6
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v366__00a6 = (v300__0080**2)
v366__00a6 = v366__00a6.reshape((1, 12))

# op _00aB_power_combination_eval
# LANG: _007H --> _00aC
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v382__00aC = (v290__007H)
v382__00aC = (v382__00aC*_00aB_coeff).reshape((1, 12))

# op _00cS_single_tensor_sum_with_axis_eval
# LANG: _00cR --> _00cT
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v458__00cT = np.sum(v457__00cR, axis = (2,)).reshape((1, 4))

# op _00cy_single_tensor_sum_with_axis_eval
# LANG: _00cx --> _00cz
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v448__00cz = np.sum(v447__00cx, axis = (2,)).reshape((1, 4))

# op _00db_single_tensor_sum_with_axis_eval
# LANG: _00da --> _00dc
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v468__00dc = np.sum(v467__00da, axis = (2,)).reshape((1, 4))

# op _00dt_power_combination_eval
# LANG: _00ds --> _00du
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v477__00du = (v476__00ds**2)
v477__00du = v477__00du.reshape((1, 4, 3))

# op _008B_power_combination_eval
# LANG: _0088 --> _008C
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v319__008C = (v304__0088**2)
v319__008C = v319__008C.reshape((1, 12))

# op _008L_linear_combination_eval
# LANG: _008G, _008K --> _008M
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v324__008M = v321__008G+-1*v323__008K

# op _008h_linear_combination_eval
# LANG: _008g --> _008i
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v309__008i = _008h_constant+v308__008g

# op _008r_linear_combination_eval
# LANG: _008q --> _008s
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v314__008s = _008r_constant+v313__008q

# op _008z_power_combination_eval
# LANG: _008a, _008c --> _008A
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v318__008A = (v305__008a)*(v306__008c)
v318__008A = v318__008A.reshape((1, 12))

# op _009I_linear_combination_eval
# LANG: _009D, _009H --> _009J
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v354__009J = v351__009D+-1*v353__009H

# op _009e_linear_combination_eval
# LANG: _009d --> _009f
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v339__009f = _009e_constant+v338__009d

# op _009o_linear_combination_eval
# LANG: _009n --> _009p
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v344__009p = _009o_constant+v343__009n

# op _009w_power_combination_eval
# LANG: _0097, _0099 --> _009x
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v348__009x = (v335__0097)*(v336__0099)
v348__009x = v348__009x.reshape((1, 12))

# op _009y_power_combination_eval
# LANG: _0095 --> _009z
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v349__009z = (v334__0095**2)
v349__009z = v349__009z.reshape((1, 12))

# op _00a1_single_tensor_sum_with_axis_eval
# LANG: _00a0 --> _00a2
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v364__00a2 = np.sum(v363__00a0, axis = (2,)).reshape((1, 12))

# op _00a9_linear_combination_eval
# LANG: _00a4 --> _00aa
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v368__00aa = _00a9_constant+v365__00a4

# op _00aD_power_combination_eval
# LANG: _0080, _00aC --> _00aE
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v383__00aE = (v382__00aC)*(v300__0080)
v383__00aE = v383__00aE.reshape((1, 12))

# op _00aX_power_combination_eval
# LANG: _007V, _006Y --> _00aY
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v393__00aY = (v297__007V)*(v267__006Y)
v393__00aY = v393__00aY.reshape((1, 12, 3))

# op _00aj_linear_combination_eval
# LANG: _00a6 --> _00ak
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v373__00ak = _00aj_constant+v366__00a6

# op _00az_linear_combination_eval
# LANG: _00a4, _00a6 --> _00aA
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v381__00aA = v365__00a4+v366__00a6

# op _00b0_power_combination_eval
# LANG: _0080 --> _00b1
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v395__00b1 = (v300__0080**2)
v395__00b1 = v395__00b1.reshape((1, 12))

# op _00b2_power_combination_eval
# LANG: _0073 --> _00b3
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v396__00b3 = (v270__0073**2)
v396__00b3 = v396__00b3.reshape((1, 12))

# op _00by_power_combination_eval
# LANG: _0080 --> _00bz
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v412__00bz = (v300__0080)
v412__00bz = (v412__00bz*_00by_coeff).reshape((1, 12))

# op _00cA_power_combination_eval
# LANG: _00cz --> _00cB
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v449__00cB = (v448__00cz**0.5)
v449__00cB = v449__00cB.reshape((1, 4))

# op _00cU_power_combination_eval
# LANG: _00cT --> _00cV
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v459__00cV = (v458__00cT**0.5)
v459__00cV = v459__00cV.reshape((1, 4))

# op _00dD_power_combination_eval
# LANG: _00cv, _00cP --> _00dE
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v482__00dE = (v446__00cv)*(v456__00cP)
v482__00dE = v482__00dE.reshape((1, 4, 3))

# op _00dd_power_combination_eval
# LANG: _00dc --> _00de
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v469__00de = (v468__00dc**0.5)
v469__00de = v469__00de.reshape((1, 4))

# op _00dv_single_tensor_sum_with_axis_eval
# LANG: _00du --> _00dw
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v478__00dw = np.sum(v477__00du, axis = (2,)).reshape((1, 4))

# op _00e2_power_combination_eval
# LANG: _00d8, _00cP --> _00e3
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v495__00e3 = (v456__00cP)*(v466__00d8)
v495__00e3 = v495__00e3.reshape((1, 4, 3))

# op _008D_linear_combination_eval
# LANG: _008A, _008C --> _008E
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v320__008E = v318__008A+-1*v319__008C

# op _008N_power_combination_eval
# LANG: _008M --> _008O
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v325__008O = (v324__008M)
v325__008O = (v325__008O*_008N_coeff).reshape((1, 12))

# op _008d_linear_combination_eval
# LANG: _008a, _0088 --> _008e
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v307__008e = v305__008a+-1*v304__0088

# op _008j_power_combination_eval
# LANG: _008i --> _008k
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v310__008k = (v309__008i**0.5)
v310__008k = v310__008k.reshape((1, 12))

# op _008n_linear_combination_eval
# LANG: _008c, _0088 --> _008o
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v312__008o = v306__008c+-1*v304__0088

# op _008t_power_combination_eval
# LANG: _008s --> _008u
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v315__008u = (v314__008s**0.5)
v315__008u = v315__008u.reshape((1, 12))

# op _009A_linear_combination_eval
# LANG: _009x, _009z --> _009B
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v350__009B = v348__009x+-1*v349__009z

# op _009K_power_combination_eval
# LANG: _009J --> _009L
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v355__009L = (v354__009J)
v355__009L = (v355__009L*_009K_coeff).reshape((1, 12))

# op _009a_linear_combination_eval
# LANG: _0097, _0095 --> _009b
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v337__009b = v335__0097+-1*v334__0095

# op _009g_power_combination_eval
# LANG: _009f --> _009h
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v340__009h = (v339__009f**0.5)
v340__009h = v340__009h.reshape((1, 12))

# op _009k_linear_combination_eval
# LANG: _0099, _0095 --> _009l
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v342__009l = v336__0099+-1*v334__0095

# op _009q_power_combination_eval
# LANG: _009p --> _009r
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v345__009r = (v344__009p**0.5)
v345__009r = v345__009r.reshape((1, 12))

# op _00aF_linear_combination_eval
# LANG: _00aA, _00aE --> _00aG
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v384__00aG = v381__00aA+-1*v383__00aE

# op _00aZ_single_tensor_sum_with_axis_eval
# LANG: _00aY --> _00a_
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v394__00a_ = np.sum(v393__00aY, axis = (2,)).reshape((1, 12))

# op _00ab_linear_combination_eval
# LANG: _00aa --> _00ac
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v369__00ac = _00ab_constant+v368__00aa

# op _00al_linear_combination_eval
# LANG: _00ak --> _00am
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v374__00am = _00al_constant+v373__00ak

# op _00at_power_combination_eval
# LANG: _00a4, _00a6 --> _00au
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v378__00au = (v365__00a4)*(v366__00a6)
v378__00au = v378__00au.reshape((1, 12))

# op _00av_power_combination_eval
# LANG: _00a2 --> _00aw
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v379__00aw = (v364__00a2**2)
v379__00aw = v379__00aw.reshape((1, 12))

# op _00b6_linear_combination_eval
# LANG: _00b1 --> _00b7
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v398__00b7 = _00b6_constant+v395__00b1

# op _00bA_power_combination_eval
# LANG: _00bz, _0073 --> _00bB
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v413__00bB = (v412__00bz)*(v270__0073)
v413__00bB = v413__00bB.reshape((1, 12))

# op _00bg_linear_combination_eval
# LANG: _00b3 --> _00bh
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v403__00bh = _00bg_constant+v396__00b3

# op _00bw_linear_combination_eval
# LANG: _00b1, _00b3 --> _00bx
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v411__00bx = v395__00b1+v396__00b3

# op _00dF_single_tensor_sum_with_axis_eval
# LANG: _00dE --> _00dG
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v483__00dG = np.sum(v482__00dE, axis = (2,)).reshape((1, 4))

# op _00dH_power_combination_eval
# LANG: _00cB, _00cV --> _00dI
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v484__00dI = (v449__00cB)*(v459__00cV)
v484__00dI = v484__00dI.reshape((1, 4))

# op _00dx_power_combination_eval
# LANG: _00dw --> _00dy
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v479__00dy = (v478__00dw**0.5)
v479__00dy = v479__00dy.reshape((1, 4))

# op _00e4_single_tensor_sum_with_axis_eval
# LANG: _00e3 --> _00e5
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v496__00e5 = np.sum(v495__00e3, axis = (2,)).reshape((1, 4))

# op _00e6_power_combination_eval
# LANG: _00de, _00cV --> _00e7
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v497__00e7 = (v459__00cV)*(v469__00de)
v497__00e7 = v497__00e7.reshape((1, 4))

# op _00es_power_combination_eval
# LANG: _00ds, _00d8 --> _00et
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v508__00et = (v466__00d8)*(v476__00ds)
v508__00et = v508__00et.reshape((1, 4, 3))

# op _0063_decompose_eval
# LANG: wing --> _0069, _0064, _0065, _0068
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v233__0064 = ((v232_wing.flatten())[src_indices__0064__0063]).reshape((1, 1, 2, 3))
v234__0065 = ((v232_wing.flatten())[src_indices__0065__0063]).reshape((1, 1, 2, 3))
v236__0068 = ((v232_wing.flatten())[src_indices__0068__0063]).reshape((1, 1, 2, 3))
v237__0069 = ((v232_wing.flatten())[src_indices__0069__0063]).reshape((1, 1, 2, 3))

# op _008P_linear_combination_eval
# LANG: _008E, _008O --> _008Q
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v326__008Q = v320__008E+v325__008O

# op _008l_power_combination_eval
# LANG: _008e, _008k --> _008m
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v311__008m = (v307__008e)*(v310__008k**-1)
v311__008m = v311__008m.reshape((1, 12))

# op _008v_power_combination_eval
# LANG: _008o, _008u --> _008w
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v316__008w = (v312__008o)*(v315__008u**-1)
v316__008w = v316__008w.reshape((1, 12))

# op _009M_linear_combination_eval
# LANG: _009B, _009L --> _009N
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v356__009N = v350__009B+v355__009L

# op _009i_power_combination_eval
# LANG: _009b, _009h --> _009j
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v341__009j = (v337__009b)*(v340__009h**-1)
v341__009j = v341__009j.reshape((1, 12))

# op _009s_power_combination_eval
# LANG: _009l, _009r --> _009t
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v346__009t = (v342__009l)*(v345__009r**-1)
v346__009t = v346__009t.reshape((1, 12))

# op _00a7_linear_combination_eval
# LANG: _00a4, _00a2 --> _00a8
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v367__00a8 = v365__00a4+-1*v364__00a2

# op _00aH_power_combination_eval
# LANG: _00aG --> _00aI
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v385__00aI = (v384__00aG)
v385__00aI = (v385__00aI*_00aH_coeff).reshape((1, 12))

# op _00ad_power_combination_eval
# LANG: _00ac --> _00ae
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v370__00ae = (v369__00ac**0.5)
v370__00ae = v370__00ae.reshape((1, 12))

# op _00ah_linear_combination_eval
# LANG: _00a6, _00a2 --> _00ai
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v372__00ai = v366__00a6+-1*v364__00a2

# op _00an_power_combination_eval
# LANG: _00am --> _00ao
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v375__00ao = (v374__00am**0.5)
v375__00ao = v375__00ao.reshape((1, 12))

# op _00ax_linear_combination_eval
# LANG: _00au, _00aw --> _00ay
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v380__00ay = v378__00au+-1*v379__00aw

# op _00b8_linear_combination_eval
# LANG: _00b7 --> _00b9
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v399__00b9 = _00b8_constant+v398__00b7

# op _00bC_linear_combination_eval
# LANG: _00bx, _00bB --> _00bD
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v414__00bD = v411__00bx+-1*v413__00bB

# op _00bi_linear_combination_eval
# LANG: _00bh --> _00bj
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v404__00bj = _00bi_constant+v403__00bh

# op _00bq_power_combination_eval
# LANG: _00b1, _00b3 --> _00br
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v408__00br = (v395__00b1)*(v396__00b3)
v408__00br = v408__00br.reshape((1, 12))

# op _00bs_power_combination_eval
# LANG: _00a_ --> _00bt
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v409__00bt = (v394__00a_**2)
v409__00bt = v409__00bt.reshape((1, 12))

# op _00dJ_linear_combination_eval
# LANG: _00dI, _00dG --> _00dK
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v485__00dK = v484__00dI+v483__00dG

# op _00dN_power_combination_eval
# LANG: _00cB --> _00dO
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v487__00dO = (v449__00cB**-1)
v487__00dO = v487__00dO.reshape((1, 4))

# op _00dP_power_combination_eval
# LANG: _00cV --> _00dQ
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v488__00dQ = (v459__00cV**-1)
v488__00dQ = v488__00dQ.reshape((1, 4))

# op _00e8_linear_combination_eval
# LANG: _00e7, _00e5 --> _00e9
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v498__00e9 = v497__00e7+v496__00e5

# op _00eS_power_combination_eval
# LANG: _00ds, _00cv --> _00eT
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v521__00eT = (v476__00ds)*(v446__00cv)
v521__00eT = v521__00eT.reshape((1, 4, 3))

# op _00ec_power_combination_eval
# LANG: _00cV --> _00ed
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v500__00ed = (v459__00cV**-1)
v500__00ed = v500__00ed.reshape((1, 4))

# op _00ee_power_combination_eval
# LANG: _00de --> _00ef
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v501__00ef = (v469__00de**-1)
v501__00ef = v501__00ef.reshape((1, 4))

# op _00eu_single_tensor_sum_with_axis_eval
# LANG: _00et --> _00ev
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v509__00ev = np.sum(v508__00et, axis = (2,)).reshape((1, 4))

# op _00ew_power_combination_eval
# LANG: _00dy, _00de --> _00ex
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v510__00ex = (v469__00de)*(v479__00dy)
v510__00ex = v510__00ex.reshape((1, 4))

# op _0066_linear_combination_eval
# LANG: _0064, _0065 --> _0067
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v235__0067 = v233__0064+-1*v234__0065

# op _006a_linear_combination_eval
# LANG: _0068, _0069 --> _006b
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v238__006b = v236__0068+-1*v237__0069

# op _008R_linear_combination_eval
# LANG: _008Q --> _008S
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v327__008S = _008R_constant+v326__008Q

# op _008x_linear_combination_eval
# LANG: _008m, _008w --> _008y
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v317__008y = v311__008m+v316__008w

# op _009O_linear_combination_eval
# LANG: _009N --> _009P
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v357__009P = _009O_constant+v356__009N

# op _009u_linear_combination_eval
# LANG: _009j, _009t --> _009v
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v347__009v = v341__009j+v346__009t

# op _00aJ_linear_combination_eval
# LANG: _00ay, _00aI --> _00aK
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v386__00aK = v380__00ay+v385__00aI

# op _00af_power_combination_eval
# LANG: _00a8, _00ae --> _00ag
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v371__00ag = (v367__00a8)*(v370__00ae**-1)
v371__00ag = v371__00ag.reshape((1, 12))

# op _00ap_power_combination_eval
# LANG: _00ai, _00ao --> _00aq
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v376__00aq = (v372__00ai)*(v375__00ao**-1)
v376__00aq = v376__00aq.reshape((1, 12))

# op _00b4_linear_combination_eval
# LANG: _00b1, _00a_ --> _00b5
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v397__00b5 = v395__00b1+-1*v394__00a_

# op _00bE_power_combination_eval
# LANG: _00bD --> _00bF
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v415__00bF = (v414__00bD)
v415__00bF = (v415__00bF*_00bE_coeff).reshape((1, 12))

# op _00ba_power_combination_eval
# LANG: _00b9 --> _00bb
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v400__00bb = (v399__00b9**0.5)
v400__00bb = v400__00bb.reshape((1, 12))

# op _00be_linear_combination_eval
# LANG: _00b3, _00a_ --> _00bf
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v402__00bf = v396__00b3+-1*v394__00a_

# op _00bk_power_combination_eval
# LANG: _00bj --> _00bl
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v405__00bl = (v404__00bj**0.5)
v405__00bl = v405__00bl.reshape((1, 12))

# op _00bu_linear_combination_eval
# LANG: _00br, _00bt --> _00bv
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v410__00bv = v408__00br+-1*v409__00bt

# op _00dL_power_combination_eval
# LANG: _00dK --> _00dM
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v486__00dM = (v485__00dK**-1)
v486__00dM = v486__00dM.reshape((1, 4))

# op _00dR_linear_combination_eval
# LANG: _00dO, _00dQ --> _00dS
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v489__00dS = v487__00dO+v488__00dQ

# op _00eC_power_combination_eval
# LANG: _00de --> _00eD
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v513__00eD = (v469__00de**-1)
v513__00eD = v513__00eD.reshape((1, 4))

# op _00eE_power_combination_eval
# LANG: _00dy --> _00eF
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v514__00eF = (v479__00dy**-1)
v514__00eF = v514__00eF.reshape((1, 4))

# op _00eU_single_tensor_sum_with_axis_eval
# LANG: _00eT --> _00eV
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v522__00eV = np.sum(v521__00eT, axis = (2,)).reshape((1, 4))

# op _00eW_power_combination_eval
# LANG: _00cB, _00dy --> _00eX
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v523__00eX = (v479__00dy)*(v449__00cB)
v523__00eX = v523__00eX.reshape((1, 4))

# op _00ea_power_combination_eval
# LANG: _00e9 --> _00eb
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v499__00eb = (v498__00e9**-1)
v499__00eb = v499__00eb.reshape((1, 4))

# op _00eg_linear_combination_eval
# LANG: _00ed, _00ef --> _00eh
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v502__00eh = v500__00ed+v501__00ef

# op _00ey_linear_combination_eval
# LANG: _00ex, _00ev --> _00ez
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v511__00ez = v510__00ex+v509__00ev

# op _006c cross_product_eval
# LANG: _0067, _006b --> _006d
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v239__006d = np.cross(v235__0067, v238__006b, axisa = 3, axisb = 3, axisc = 3)

# op _0081 cross_product_eval
# LANG: _006Y, _007h --> _0082
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v301__0082 = np.cross(v267__006Y, v277__007h, axisa = 2, axisb = 2, axisc = 2)

# op _008T_power_combination_eval
# LANG: _008y, _008S --> _008U
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v328__008U = (v317__008y)*(v327__008S**-1)
v328__008U = v328__008U.reshape((1, 12))

# op _008Z cross_product_eval
# LANG: _007B, _007h --> _008_
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v331__008_ = np.cross(v277__007h, v287__007B, axisa = 2, axisb = 2, axisc = 2)

# op _009Q_power_combination_eval
# LANG: _009v, _009P --> _009R
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v358__009R = (v347__009v)*(v357__009P**-1)
v358__009R = v358__009R.reshape((1, 12))

# op _00aL_linear_combination_eval
# LANG: _00aK --> _00aM
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v387__00aM = _00aL_constant+v386__00aK

# op _00ar_linear_combination_eval
# LANG: _00ag, _00aq --> _00as
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v377__00as = v371__00ag+v376__00aq

# op _00bG_linear_combination_eval
# LANG: _00bv, _00bF --> _00bH
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v416__00bH = v410__00bv+v415__00bF

# op _00bc_power_combination_eval
# LANG: _00b5, _00bb --> _00bd
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v401__00bd = (v397__00b5)*(v400__00bb**-1)
v401__00bd = v401__00bd.reshape((1, 12))

# op _00bm_power_combination_eval
# LANG: _00bf, _00bl --> _00bn
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v406__00bn = (v402__00bf)*(v405__00bl**-1)
v406__00bn = v406__00bn.reshape((1, 12))

# op _00dT_power_combination_eval
# LANG: _00dM, _00dS --> _00dU
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v490__00dU = (v486__00dM)*(v489__00dS)
v490__00dU = v490__00dU.reshape((1, 4))

# op _00dZ cross_product_eval
# LANG: _00d8, _00cP --> _00d_
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v493__00d_ = np.cross(v456__00cP, v466__00d8, axisa = 2, axisb = 2, axisc = 2)

# op _00dz cross_product_eval
# LANG: _00cv, _00cP --> _00dA
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v480__00dA = np.cross(v446__00cv, v456__00cP, axisa = 2, axisb = 2, axisc = 2)

# op _00eA_power_combination_eval
# LANG: _00ez --> _00eB
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v512__00eB = (v511__00ez**-1)
v512__00eB = v512__00eB.reshape((1, 4))

# op _00eG_linear_combination_eval
# LANG: _00eD, _00eF --> _00eH
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v515__00eH = v513__00eD+v514__00eF

# op _00eY_linear_combination_eval
# LANG: _00eX, _00eV --> _00eZ
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v524__00eZ = v523__00eX+v522__00eV

# op _00ei_power_combination_eval
# LANG: _00eb, _00eh --> _00ej
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v503__00ej = (v499__00eb)*(v502__00eh)
v503__00ej = v503__00ej.reshape((1, 4))

# op _00f1_power_combination_eval
# LANG: _00dy --> _00f2
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v526__00f2 = (v479__00dy**-1)
v526__00f2 = v526__00f2.reshape((1, 4))

# op _00f3_power_combination_eval
# LANG: _00cB --> _00f4
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v527__00f4 = (v449__00cB**-1)
v527__00f4 = v527__00f4.reshape((1, 4))

# op _005F_indexed_passthrough_eval
# LANG: p, q, r --> ang_vel
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v218_ang_vel__temp[i_v215_p__005F_indexed_passthrough_eval] = v215_p.flatten()
v218_ang_vel = v218_ang_vel__temp.copy()
v218_ang_vel__temp[i_v216_q__005F_indexed_passthrough_eval] = v216_q.flatten()
v218_ang_vel = v218_ang_vel__temp.copy()
v218_ang_vel__temp[i_v217_r__005F_indexed_passthrough_eval] = v217_r.flatten()
v218_ang_vel = v218_ang_vel__temp.copy()

# op _005I expand_array_eval
# LANG: wing_rot_ref --> _005J
# SHAPES: (1, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v221__005J = np.einsum('ad,bc->abcd', v220_wing_rot_ref.reshape((1, 3)) ,np.ones((1, 2))).reshape((1, 1, 2, 3))

# op _006e_power_combination_eval
# LANG: _006d --> _006f
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v240__006f = (v239__006d**2)
v240__006f = v240__006f.reshape((1, 1, 2, 3))

# op _0083_power_combination_eval
# LANG: _0082 --> _0084
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v302__0084 = (v301__0082)
v302__0084 = (v302__0084*_0083_coeff).reshape((1, 12, 3))

# op _008V expand_array_eval
# LANG: _008U --> _008W
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v329__008W = np.einsum('ab,c->abc', v328__008U.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _0090_power_combination_eval
# LANG: _008_ --> _0091
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v332__0091 = (v331__008_)
v332__0091 = (v332__0091*_0090_coeff).reshape((1, 12, 3))

# op _009S expand_array_eval
# LANG: _009R --> _009T
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v359__009T = np.einsum('ab,c->abc', v358__009R.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _009W cross_product_eval
# LANG: _007V, _007B --> _009X
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v361__009X = np.cross(v287__007B, v297__007V, axisa = 2, axisb = 2, axisc = 2)

# op _00aN_power_combination_eval
# LANG: _00as, _00aM --> _00aO
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v388__00aO = (v377__00as)*(v387__00aM**-1)
v388__00aO = v388__00aO.reshape((1, 12))

# op _00bI_linear_combination_eval
# LANG: _00bH --> _00bJ
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v417__00bJ = _00bI_constant+v416__00bH

# op _00bo_linear_combination_eval
# LANG: _00bd, _00bn --> _00bp
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v407__00bp = v401__00bd+v406__00bn

# op _00dB_power_combination_eval
# LANG: _00dA --> _00dC
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v481__00dC = (v480__00dA)
v481__00dC = (v481__00dC*_00dB_coeff).reshape((1, 4, 3))

# op _00dV expand_array_eval
# LANG: _00dU --> _00dW
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v491__00dW = np.einsum('ab,c->abc', v490__00dU.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00e0_power_combination_eval
# LANG: _00d_ --> _00e1
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v494__00e1 = (v493__00d_)
v494__00e1 = (v494__00e1*_00e0_coeff).reshape((1, 4, 3))

# op _00eI_power_combination_eval
# LANG: _00eB, _00eH --> _00eJ
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v516__00eJ = (v512__00eB)*(v515__00eH)
v516__00eJ = v516__00eJ.reshape((1, 4))

# op _00e__power_combination_eval
# LANG: _00eZ --> _00f0
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v525__00f0 = (v524__00eZ**-1)
v525__00f0 = v525__00f0.reshape((1, 4))

# op _00ek expand_array_eval
# LANG: _00ej --> _00el
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v504__00el = np.einsum('ab,c->abc', v503__00ej.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00eo cross_product_eval
# LANG: _00ds, _00d8 --> _00ep
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v506__00ep = np.cross(v466__00d8, v476__00ds, axisa = 2, axisb = 2, axisc = 2)

# op _00f5_linear_combination_eval
# LANG: _00f2, _00f4 --> _00f6
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v528__00f6 = v526__00f2+v527__00f4

# op _005K_linear_combination_eval
# LANG: _005J, wing_coll_pts_coords --> _005L
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v222__005L = v434_wing_coll_pts_coords+-1*v221__005J

# op _005M expand_array_eval
# LANG: ang_vel --> _005N
# SHAPES: (1, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v223__005N = np.einsum('ad,bc->abcd', v218_ang_vel.reshape((1, 3)) ,np.ones((1, 2))).reshape((1, 1, 2, 3))

# op _006g_single_tensor_sum_with_axis_eval
# LANG: _006f --> _006h
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v241__006h = np.sum(v240__006f, axis = (3,)).reshape((1, 1, 2))

# op _008X_power_combination_eval
# LANG: _008W, _0084 --> _008Y
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v330__008Y = (v329__008W)*(v302__0084)
v330__008Y = v330__008Y.reshape((1, 12, 3))

# op _009U_power_combination_eval
# LANG: _009T, _0091 --> _009V
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v360__009V = (v359__009T)*(v332__0091)
v360__009V = v360__009V.reshape((1, 12, 3))

# op _009Y_power_combination_eval
# LANG: _009X --> _009Z
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v362__009Z = (v361__009X)
v362__009Z = (v362__009Z*_009Y_coeff).reshape((1, 12, 3))

# op _00aP expand_array_eval
# LANG: _00aO --> _00aQ
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v389__00aQ = np.einsum('ab,c->abc', v388__00aO.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _00aT cross_product_eval
# LANG: _007V, _006Y --> _00aU
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v391__00aU = np.cross(v297__007V, v267__006Y, axisa = 2, axisb = 2, axisc = 2)

# op _00bK_power_combination_eval
# LANG: _00bp, _00bJ --> _00bL
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v418__00bL = (v407__00bp)*(v417__00bJ**-1)
v418__00bL = v418__00bL.reshape((1, 12))

# op _00dX_power_combination_eval
# LANG: _00dW, _00dC --> _00dY
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v492__00dY = (v491__00dW)*(v481__00dC)
v492__00dY = v492__00dY.reshape((1, 4, 3))

# op _00eK expand_array_eval
# LANG: _00eJ --> _00eL
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v517__00eL = np.einsum('ab,c->abc', v516__00eJ.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00eO cross_product_eval
# LANG: _00ds, _00cv --> _00eP
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v519__00eP = np.cross(v476__00ds, v446__00cv, axisa = 2, axisb = 2, axisc = 2)

# op _00em_power_combination_eval
# LANG: _00el, _00e1 --> _00en
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v505__00en = (v504__00el)*(v494__00e1)
v505__00en = v505__00en.reshape((1, 4, 3))

# op _00eq_power_combination_eval
# LANG: _00ep --> _00er
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v507__00er = (v506__00ep)
v507__00er = (v507__00er*_00eq_coeff).reshape((1, 4, 3))

# op _00f7_power_combination_eval
# LANG: _00f0, _00f6 --> _00f8
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v529__00f8 = (v525__00f0)*(v528__00f6)
v529__00f8 = v529__00f8.reshape((1, 4))

# op _005O cross_product_eval
# LANG: _005N, _005L --> _005P
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v224__005P = np.cross(v223__005N, v222__005L, axisa = 3, axisb = 3, axisc = 3)

# op _006i_power_combination_eval
# LANG: _006h --> _006j
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v242__006j = (v241__006h**0.5)
v242__006j = v242__006j.reshape((1, 1, 2))

# op _00aR_power_combination_eval
# LANG: _00aQ, _009Z --> _00aS
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v390__00aS = (v389__00aQ)*(v362__009Z)
v390__00aS = v390__00aS.reshape((1, 12, 3))

# op _00aV_power_combination_eval
# LANG: _00aU --> _00aW
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v392__00aW = (v391__00aU)
v392__00aW = (v392__00aW*_00aV_coeff).reshape((1, 12, 3))

# op _00bM expand_array_eval
# LANG: _00bL --> _00bN
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v419__00bN = np.einsum('ab,c->abc', v418__00bL.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _00bQ_linear_combination_eval
# LANG: _008Y, _009V --> _00bR
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v421__00bR = v330__008Y+v360__009V

# op _00eM_power_combination_eval
# LANG: _00eL, _00er --> _00eN
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v518__00eN = (v517__00eL)*(v507__00er)
v518__00eN = v518__00eN.reshape((1, 4, 3))

# op _00eQ_power_combination_eval
# LANG: _00eP --> _00eR
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v520__00eR = (v519__00eP)
v520__00eR = (v520__00eR*_00eQ_coeff).reshape((1, 4, 3))

# op _00f9 expand_array_eval
# LANG: _00f8 --> _00fa
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v530__00fa = np.einsum('ab,c->abc', v529__00f8.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00fd_linear_combination_eval
# LANG: _00dY, _00en --> _00fe
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v532__00fe = v492__00dY+v505__00en

# op _005Q reshape_eval
# LANG: _005P --> _005R
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v225__005R = v224__005P.reshape((1, 2, 3))

# op _005S expand_array_eval
# LANG: frame_vel --> _005T
# SHAPES: (1, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v226__005T = np.einsum('ac,b->abc', v544_frame_vel.reshape((1, 3)) ,np.ones((2,))).reshape((1, 2, 3))

# op _006k expand_array_eval
# LANG: _006j --> _006l
# SHAPES: (1, 1, 2) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v243__006l = np.einsum('abc,d->abcd', v242__006j.reshape((1, 1, 2)) ,np.ones((3,))).reshape((1, 1, 2, 3))

# op _00bO_power_combination_eval
# LANG: _00bN, _00aW --> _00bP
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v420__00bP = (v419__00bN)*(v392__00aW)
v420__00bP = v420__00bP.reshape((1, 12, 3))

# op _00bS_linear_combination_eval
# LANG: _00bR, _00aS --> _00bT
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v422__00bT = v421__00bR+v390__00aS

# op _00fb_power_combination_eval
# LANG: _00fa, _00eR --> _00fc
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v531__00fc = (v530__00fa)*(v520__00eR)
v531__00fc = v531__00fc.reshape((1, 4, 3))

# op _00ff_linear_combination_eval
# LANG: _00fe, _00eN --> _00fg
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v533__00fg = v532__00fe+v518__00eN

# op _005V_linear_combination_eval
# LANG: _005R, _005T --> _005W
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v228__005W = v225__005R+v226__005T

# op _005X reshape_eval
# LANG: wing_coll_vel --> _005Y
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v229__005Y = v227_wing_coll_vel.reshape((1, 2, 3))

# op _006m_power_combination_eval
# LANG: _006d, _006l --> wing_bd_vtx_normals
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v537_wing_bd_vtx_normals = (v239__006d)*(v243__006l**-1)
v537_wing_bd_vtx_normals = v537_wing_bd_vtx_normals.reshape((1, 1, 2, 3))

# op _00bU_linear_combination_eval
# LANG: _00bT, _00bP --> aic_M00
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v423_aic_M00 = v422__00bT+v420__00bP

# op _00fh_linear_combination_eval
# LANG: _00fg, _00fc --> aic_bd00
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v534_aic_bd00 = v533__00fg+v531__00fc

# op _005Z_linear_combination_eval
# LANG: _005W, _005Y --> _005_
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v230__005_ = v228__005W+v229__005Y

# op _006B reshape_eval
# LANG: aic_M00 --> _006C
# SHAPES: (1, 12, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v254__006C = v423_aic_M00.reshape((1, 2, 6, 3))

# op _00bZ reshape_eval
# LANG: wing_bd_vtx_normals --> _00b_
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v427__00b_ = v537_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _00c8 reshape_eval
# LANG: aic_bd00 --> _00c9
# SHAPES: (1, 4, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v433__00c9 = v534_aic_bd00.reshape((1, 2, 2, 3))

# op _00fm reshape_eval
# LANG: wing_bd_vtx_normals --> _00fn
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v538__00fn = v537_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _0060_linear_combination_eval
# LANG: _005_ --> wing_kinematic_vel
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v246_wing_kinematic_vel = -1*v230__005_

# op _006D_indexed_passthrough_eval
# LANG: _006C --> aic_M
# SHAPES: (1, 2, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v425_aic_M__temp[i_v254__006C__006D_indexed_passthrough_eval] = v254__006C.flatten()
v425_aic_M = v425_aic_M__temp.copy()

# op _006r reshape_eval
# LANG: wing_bd_vtx_normals --> _006s
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v248__006s = v537_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _00c0_indexed_passthrough_eval
# LANG: _00b_ --> normal_concatenated_M_mat
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v424_normal_concatenated_M_mat__temp[i_v427__00b___00c0_indexed_passthrough_eval] = v427__00b_.flatten()
v424_normal_concatenated_M_mat = v424_normal_concatenated_M_mat__temp.copy()

# op _00ca_indexed_passthrough_eval
# LANG: _00c9 --> aic_bd
# SHAPES: (1, 2, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v536_aic_bd__temp[i_v433__00c9__00ca_indexed_passthrough_eval] = v433__00c9.flatten()
v536_aic_bd = v536_aic_bd__temp.copy()

# op _00fo_indexed_passthrough_eval
# LANG: _00fn --> normal_concatenated_aic_bd_proj
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v535_normal_concatenated_aic_bd_proj__temp[i_v538__00fn__00fo_indexed_passthrough_eval] = v538__00fn.flatten()
v535_normal_concatenated_aic_bd_proj = v535_normal_concatenated_aic_bd_proj__temp.copy()

# op _006t_custom_explicit_eval
# LANG: _006s, wing_kinematic_vel --> b
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
temp = _006t_custom_explicit_func_b.solve(v246_wing_kinematic_vel, v248__006s)
v249_b = temp[0].copy()

# op _00c1_custom_explicit_eval
# LANG: normal_concatenated_M_mat, aic_M --> M_mat
# SHAPES: (1, 2, 3), (1, 2, 6, 3) --> (1, 2, 6)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
temp = _00c1_custom_explicit_func_M_mat.solve(v425_aic_M, v424_normal_concatenated_M_mat)
v428_M_mat = temp[0].copy()

# op _00fp_custom_explicit_eval
# LANG: normal_concatenated_aic_bd_proj, aic_bd --> aic_bd_proj
# SHAPES: (1, 2, 3), (1, 2, 2, 3) --> (1, 2, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
temp = _00fp_custom_explicit_func_aic_bd_proj.solve(v536_aic_bd, v535_normal_concatenated_aic_bd_proj)
v539_aic_bd_proj = temp[0].copy()

# op _004V_indexed_passthrough_eval
# LANG: wing_gamma_w --> gamma_w
# SHAPES: (1, 3, 2) --> (1, 3, 2)
# full namespace: combine_gamma_w
v202_gamma_w__temp[i_v190_wing_gamma_w__004V_indexed_passthrough_eval] = v190_wing_gamma_w.flatten()
v202_gamma_w = v202_gamma_w__temp.copy()

# op _005g_newton_implict_eval
# LANG: gamma_w, b, aic_bd_proj, M_mat --> gamma_b
# SHAPES: (1, 3, 2), (1, 2), (1, 2, 2), (1, 2, 6) --> (1, 2)
# full namespace: solve_gamma_b_group
_005g_newton.set_guess(initial_guess_v540_gamma_b)
_005g_newton_out = _005g_newton.solve(v539_aic_bd_proj, v428_M_mat, v202_gamma_w, v249_b)
v540_gamma_b = _005g_newton_out[0]

# op _00fz_linear_combination_eval
# LANG: frame_vel --> _00fA
# SHAPES: (1, 3) --> (1, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v545__00fA = -1*v544_frame_vel

# op _00fB expand_array_eval
# LANG: _00fA --> wing_wake_kinematic_vel
# SHAPES: (1, 3) --> (1, 3, 3, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v546_wing_wake_kinematic_vel = np.einsum('ad,bc->abcd', v545__00fA.reshape((1, 3)) ,np.ones((3, 3))).reshape((1, 3, 3, 3))

# op _00fw_linear_combination_eval
# LANG: wing_wake_kinematic_vel --> wing_wake_total_vel
# SHAPES: (1, 3, 3, 3) --> (1, 3, 3, 3)
# full namespace: ComputeWakeTotalVel
v543_wing_wake_total_vel = v546_wing_wake_kinematic_vel

# op _0016_decompose_eval
# LANG: wing_wake_total_vel --> _001P, _0017
# SHAPES: (1, 3, 3, 3) --> (1, 2, 3, 3), (1, 1, 3, 3)
# full namespace: 
v53__0017 = ((v543_wing_wake_total_vel.flatten())[src_indices__0017__0016]).reshape((1, 1, 3, 3))
v76__001P = ((v543_wing_wake_total_vel.flatten())[src_indices__001P__0016]).reshape((1, 2, 3, 3))

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
v52__0015 = ((v435_wing_bd_vtx_coords.flatten())[src_indices__0015__0014]).reshape((1, 1, 3, 3))

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
v57__001f = ((v208_wing_wake_coords.flatten())[src_indices__001f__001e]).reshape((1, 1, 3, 3))
v72__001J = ((v208_wing_wake_coords.flatten())[src_indices__001J__001e]).reshape((1, 2, 3, 3))
v74__001M = ((v208_wing_wake_coords.flatten())[src_indices__001M__001e]).reshape((1, 2, 3, 3))

# op _003m_linear_combination_eval
# LANG: _002H, _002K --> _003n
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v135__003n = v110__002H+-1*v112__002K

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

# op _002D_linear_combination_eval
# LANG: wing --> _002E
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: MeshPreprocessing_comp
v108__002E = v232_wing

# op _003o pnorm_axis_eval
# LANG: _003n --> _003p
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3)
# full namespace: MeshPreprocessing_comp
v136__003p = np.sum(v135__003n**2,axis=(3,))**(1 / 2)

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
v93__002c = (v116_w**2)
v93__002c = v93__002c.reshape((1, 1))

# op _003K_decompose_eval
# LANG: _002E --> _003Q, _003L, _003M, _003P
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v149__003L = ((v108__002E.flatten())[src_indices__003L__003K]).reshape((1, 1, 2, 3))
v150__003M = ((v108__002E.flatten())[src_indices__003M__003K]).reshape((1, 1, 2, 3))
v152__003P = ((v108__002E.flatten())[src_indices__003P__003K]).reshape((1, 1, 2, 3))
v153__003Q = ((v108__002E.flatten())[src_indices__003Q__003K]).reshape((1, 1, 2, 3))

# op _003q_decompose_eval
# LANG: _003p --> _003s, _003r
# SHAPES: (1, 1, 3) --> (1, 1, 2), (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v137__003r = ((v136__003p.flatten())[src_indices__003r__003q]).reshape((1, 1, 2))
v138__003s = ((v136__003p.flatten())[src_indices__003s__003q]).reshape((1, 1, 2))

# op _0044_power_combination_eval
# LANG: _0043 --> _0045
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v162__0045 = (v161__0043)
v162__0045 = (v162__0045*_0044_coeff).reshape((1, 1, 2, 3))

# op _0047_power_combination_eval
# LANG: _0046 --> _0048
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v164__0048 = (v163__0046)
v164__0048 = (v164__0048*_0047_coeff).reshape((1, 1, 2, 3))

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
v181_v_inf_sq = v92__002a+v93__002c

# op _003N_linear_combination_eval
# LANG: _003L, _003M --> _003O
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v151__003O = v149__003L+-1*v150__003M

# op _003R_linear_combination_eval
# LANG: _003P, _003Q --> _003S
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v154__003S = v152__003P+-1*v153__003Q

# op _003t_linear_combination_eval
# LANG: _003r, _003s --> _003u
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v139__003u = v137__003r+v138__003s

# op _0049_linear_combination_eval
# LANG: _0045, _0048 --> _004a
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v165__004a = v162__0045+v164__0048

# op _004c_power_combination_eval
# LANG: _004b --> _004d
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v167__004d = (v166__004b)
v167__004d = (v167__004d*_004c_coeff).reshape((1, 1, 2, 3))

# op _00fs_decompose_eval
# LANG: gamma_b --> wing_gamma_b
# SHAPES: (1, 2) --> (1, 2)
# full namespace: seperate_gamma_b
v541_wing_gamma_b = ((v540_gamma_b.flatten())[src_indices_wing_gamma_b__00fs]).reshape((1, 2))

# op _000I_decompose_eval
# LANG: wing_gamma_b --> _000J
# SHAPES: (1, 2) --> (1, 2)
# full namespace: 
v38__000J = ((v541_wing_gamma_b.flatten())[src_indices__000J__000I]).reshape((1, 2))

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

# op _003T cross_product_eval
# LANG: _003O, _003S --> _003U
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v155__003U = np.cross(v151__003O, v154__003S, axisa = 3, axisb = 3, axisc = 3)

# op _003v_power_combination_eval
# LANG: _003u --> wing_chord_length
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v140_wing_chord_length = (v139__003u)
v140_wing_chord_length = (v140_wing_chord_length*_003v_coeff).reshape((1, 1, 2))

# op _003z_linear_combination_eval
# LANG: _003x, _003y --> _003A
# SHAPES: (1, 2, 2, 3), (1, 2, 2, 3) --> (1, 2, 2, 3)
# full namespace: MeshPreprocessing_comp
v143__003A = v141__003x+-1*v142__003y

# op _004J_power_combination_eval
# LANG: v_inf_sq --> _004K
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v184__004K = (v181_v_inf_sq**0.5)
v184__004K = v184__004K.reshape((1, 1))

# op _004e_linear_combination_eval
# LANG: _004a, _004d --> _004f
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v168__004f = v165__004a+v167__004d

# op _004g_power_combination_eval
# LANG: _003f --> _004h
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v169__004h = (v131__003f)
v169__004h = (v169__004h*_004g_coeff).reshape((1, 1, 2, 3))

# op _000K reshape_eval
# LANG: _000J --> _000L
# SHAPES: (1, 2) --> (1, 1, 2)
# full namespace: 
v39__000L = v38__000J.reshape((1, 1, 2))

# op _000M_decompose_eval
# LANG: wing_gamma_w --> _000U, _000N, _000T
# SHAPES: (1, 3, 2) --> (1, 2, 2), (1, 1, 2), (1, 2, 2)
# full namespace: 
v40__000N = ((v190_wing_gamma_w.flatten())[src_indices__000N__000M]).reshape((1, 1, 2))
v43__000T = ((v190_wing_gamma_w.flatten())[src_indices__000T__000M]).reshape((1, 2, 2))
v44__000U = ((v190_wing_gamma_w.flatten())[src_indices__000U__000M]).reshape((1, 2, 2))

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

# op _003B pnorm_axis_eval
# LANG: _003A --> _003C
# SHAPES: (1, 2, 2, 3) --> (1, 2, 2)
# full namespace: MeshPreprocessing_comp
v144__003C = np.sum(v143__003A**2,axis=(3,))**(1 / 2)

# op _003V_power_combination_eval
# LANG: _003U --> _003W
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v156__003W = (v155__003U**2)
v156__003W = v156__003W.reshape((1, 1, 2, 3))

# op _004F_single_tensor_sum_with_axis_eval
# LANG: wing_chord_length --> _004G
# SHAPES: (1, 1, 2) --> (1, 2)
# full namespace: MeshPreprocessing_comp
v182__004G = np.sum(v140_wing_chord_length, axis = (1,)).reshape((1, 2))

# op _004L_power_combination_eval
# LANG: density, _004K --> _004M
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v185__004M = (v180_density)*(v184__004K)
v185__004M = v185__004M.reshape((1, 1))

# op _004i_linear_combination_eval
# LANG: _004f, _004h --> _004j
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v170__004j = v168__004f+v169__004h

# op _004p_power_combination_eval
# LANG: _0043 --> _004q
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v173__004q = (v161__0043)
v173__004q = (v173__004q*_004p_coeff).reshape((1, 1, 2, 3))

# op _004r_power_combination_eval
# LANG: _004b --> _004s
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v174__004s = (v166__004b)
v174__004s = (v174__004s*_004r_coeff).reshape((1, 1, 2, 3))

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

# op _003D_decompose_eval
# LANG: _003C --> _003F, _003E
# SHAPES: (1, 2, 2) --> (1, 1, 2), (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v145__003E = ((v144__003C.flatten())[src_indices__003E__003D]).reshape((1, 1, 2))
v146__003F = ((v144__003C.flatten())[src_indices__003F__003D]).reshape((1, 1, 2))

# op _003X_single_tensor_sum_with_axis_eval
# LANG: _003W --> _003Y
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v157__003Y = np.sum(v156__003W, axis = (3,)).reshape((1, 1, 2))

# op _004H reshape_eval
# LANG: _004G --> _004I
# SHAPES: (1, 2) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v183__004I = v182__004G.reshape((1, 2, 1))

# op _004N expand_array_eval
# LANG: _004M --> _004O
# SHAPES: (1, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v186__004O = np.einsum('ac,b->abc', v185__004M.reshape((1, 1)) ,np.ones((2,))).reshape((1, 2, 1))

# op _004k reshape_eval
# LANG: _004j --> _004l
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v171__004l = v170__004j.reshape((1, 2, 3))

# op _004t_linear_combination_eval
# LANG: _004q, _004s --> _004u
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v175__004u = v173__004q+v174__004s

# op _004v_power_combination_eval
# LANG: _0046 --> _004w
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v176__004w = (v163__0046)
v176__004w = (v176__004w*_004v_coeff).reshape((1, 1, 2, 3))

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

# op _003G_linear_combination_eval
# LANG: _003E, _003F --> _003H
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v147__003H = v145__003E+v146__003F

# op _003Z_power_combination_eval
# LANG: _003Y --> _003_
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v158__003_ = (v157__003Y**0.5)
v158__003_ = v158__003_.reshape((1, 1, 2))

# op _004P_power_combination_eval
# LANG: _004O, _004I --> _004Q
# SHAPES: (1, 2, 1), (1, 2, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v187__004Q = (v186__004O)*(v183__004I)
v187__004Q = v187__004Q.reshape((1, 2, 1))

# op _004m_linear_combination_eval
# LANG: _004l --> _004n
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v172__004n = -1*v171__004l

# op _004x_linear_combination_eval
# LANG: _004u, _004w --> _004y
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v177__004y = v175__004u+v176__004w

# op _004z_power_combination_eval
# LANG: _003f --> _004A
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v178__004A = (v131__003f)
v178__004A = (v178__004A*_004z_coeff).reshape((1, 1, 2, 3))

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

# op _003I_power_combination_eval
# LANG: _003H --> wing_span_length
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v148_wing_span_length = (v147__003H)
v148_wing_span_length = (v148_wing_span_length*_003I_coeff).reshape((1, 1, 2))

# op _0040_power_combination_eval
# LANG: _003_ --> wing_s_panel
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v159_wing_s_panel = (v158__003_)
v159_wing_s_panel = (v159_wing_s_panel*_0040_coeff).reshape((1, 1, 2))

# op _004B_linear_combination_eval
# LANG: _004y, _004A --> wing_eval_pts_coords
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v179_wing_eval_pts_coords = v177__004y+v178__004A

# op _004R_power_combination_eval
# LANG: _004Q --> wing_re_span
# SHAPES: (1, 2, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v188_wing_re_span = (v187__004Q)
v188_wing_re_span = (v188_wing_re_span*_004R_coeff).reshape((1, 2, 1))

# op _004o_indexed_passthrough_eval
# LANG: _004n --> bd_vec
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v160_bd_vec__temp[i_v172__004n__004o_indexed_passthrough_eval] = v172__004n.flatten()
v160_bd_vec = v160_bd_vec__temp.copy()

# op _006v_indexed_passthrough_eval
# LANG: _006s --> normal_concatenated_b
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v245_normal_concatenated_b__temp[i_v248__006s__006v_indexed_passthrough_eval] = v248__006s.flatten()
v245_normal_concatenated_b = v245_normal_concatenated_b__temp.copy()