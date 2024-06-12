

# RUN_MODEL_ode_system

# system evaluation block

# op _002v_linear_combination_eval
# LANG: u --> _002w
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v103__002w = -1*v79_u

# op _002y_linear_combination_eval
# LANG: w --> _002z
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v104__002z = -1*v115_w

# op _002x_indexed_passthrough_eval
# LANG: _002w, _002z --> frame_vel
# SHAPES: (1, 1), (1, 1) --> (1, 3)
# full namespace: adapter_comp
v543_frame_vel__temp[i_v103__002w__002x_indexed_passthrough_eval] = v103__002w.flatten()
v543_frame_vel = v543_frame_vel__temp.copy()
v543_frame_vel__temp[i_v104__002z__002x_indexed_passthrough_eval] = v104__002z.flatten()
v543_frame_vel = v543_frame_vel__temp.copy()

# op _002S_decompose_eval
# LANG: frame_vel --> _002X, _002T
# SHAPES: (1, 3) --> (1, 1), (1, 1)
# full namespace: MeshPreprocessing_comp
v117__002T = ((v543_frame_vel.flatten())[src_indices__002T__002S]).reshape((1, 1))
v119__002X = ((v543_frame_vel.flatten())[src_indices__002X__002S]).reshape((1, 1))

# op _002U_linear_combination_eval
# LANG: _002T --> _002V
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v118__002V = -1*v117__002T

# op _002Y_linear_combination_eval
# LANG: _002X --> _002Z
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v120__002Z = -1*v119__002X

# op _002F_decompose_eval
# LANG: wing --> _003e, _002G, _002J, _0037, _0038, _003d, _003w, _003x, _0042, _0045, _004a
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 2, 2, 3), (1, 2, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v109__002G = ((v231_wing.flatten())[src_indices__002G__002F]).reshape((1, 1, 3, 3))
v111__002J = ((v231_wing.flatten())[src_indices__002J__002F]).reshape((1, 1, 3, 3))
v125__0037 = ((v231_wing.flatten())[src_indices__0037__002F]).reshape((1, 1, 2, 3))
v126__0038 = ((v231_wing.flatten())[src_indices__0038__002F]).reshape((1, 1, 2, 3))
v129__003d = ((v231_wing.flatten())[src_indices__003d__002F]).reshape((1, 1, 2, 3))
v130__003e = ((v231_wing.flatten())[src_indices__003e__002F]).reshape((1, 1, 2, 3))
v140__003w = ((v231_wing.flatten())[src_indices__003w__002F]).reshape((1, 2, 2, 3))
v141__003x = ((v231_wing.flatten())[src_indices__003x__002F]).reshape((1, 2, 2, 3))
v160__0042 = ((v231_wing.flatten())[src_indices__0042__002F]).reshape((1, 1, 2, 3))
v162__0045 = ((v231_wing.flatten())[src_indices__0045__002F]).reshape((1, 1, 2, 3))
v165__004a = ((v231_wing.flatten())[src_indices__004a__002F]).reshape((1, 1, 2, 3))

# op _002W_indexed_passthrough_eval
# LANG: _002V, _002Z, w --> fs
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v116_fs__temp[i_v118__002V__002W_indexed_passthrough_eval] = v118__002V.flatten()
v116_fs = v116_fs__temp.copy()
v116_fs__temp[i_v120__002Z__002W_indexed_passthrough_eval] = v120__002Z.flatten()
v116_fs = v116_fs__temp.copy()
v116_fs__temp[i_v115_w__002W_indexed_passthrough_eval] = v115_w.flatten()
v116_fs = v116_fs__temp.copy()

# op _005p_decompose_eval
# LANG: wing_wake_coords --> _005q
# SHAPES: (1, 3, 3, 3) --> (1, 1, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v209__005q = ((v207_wing_wake_coords.flatten())[src_indices__005q__005p]).reshape((1, 1, 3, 3))

# op _002__power_combination_eval
# LANG: fs --> _0030
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v121__0030 = (v116_fs)
v121__0030 = (v121__0030*_002__coeff).reshape((1, 3))

# op _0039_linear_combination_eval
# LANG: _0037, _0038 --> _003a
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v127__003a = v125__0037+v126__0038

# op _003f_linear_combination_eval
# LANG: _003e, _003d --> _003g
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v131__003g = v129__003d+v130__003e

# op _005r_power_combination_eval
# LANG: _005q --> _005s
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v210__005s = (v209__005q)
v210__005s = v210__005s.reshape((1, 1, 3, 3))

# op _0031_power_combination_eval
# LANG: _0030 --> _0032
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v122__0032 = (v121__0030)
v122__0032 = (v122__0032*_0031_coeff).reshape((1, 3))

# op _003b_power_combination_eval
# LANG: _003a --> _003c
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v128__003c = (v127__003a)
v128__003c = (v128__003c*_003b_coeff).reshape((1, 1, 2, 3))

# op _003h_power_combination_eval
# LANG: _003g --> _003i
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v132__003i = (v131__003g)
v132__003i = (v132__003i*_003h_coeff).reshape((1, 1, 2, 3))

# op _005o_indexed_passthrough_eval
# LANG: _005s, wing_wake_coords --> wing_TE_wake_coords
# SHAPES: (1, 1, 3, 3), (1, 3, 3, 3) --> (1, 4, 3, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v255_wing_TE_wake_coords__temp[i_v207_wing_wake_coords__005o_indexed_passthrough_eval] = v207_wing_wake_coords.flatten()
v255_wing_TE_wake_coords = v255_wing_TE_wake_coords__temp.copy()
v255_wing_TE_wake_coords__temp[i_v210__005s__005o_indexed_passthrough_eval] = v210__005s.flatten()
v255_wing_TE_wake_coords = v255_wing_TE_wake_coords__temp.copy()

# op _002H_power_combination_eval
# LANG: _002G --> _002I
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v110__002I = (v109__002G)
v110__002I = (v110__002I*_002H_coeff).reshape((1, 1, 3, 3))

# op _002K_power_combination_eval
# LANG: _002J --> _002L
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v112__002L = (v111__002J)
v112__002L = (v112__002L*_002K_coeff).reshape((1, 1, 3, 3))

# op _0033 expand_array_eval
# LANG: _0032 --> _0034
# SHAPES: (1, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v123__0034 = np.einsum('ad,bc->abcd', v122__0032.reshape((1, 3)) ,np.ones((1, 3))).reshape((1, 1, 3, 3))

# op _003j_linear_combination_eval
# LANG: _003c, _003i --> wing_coll_pts_coords
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v433_wing_coll_pts_coords = v128__003c+v132__003i

# op _006F_decompose_eval
# LANG: wing_TE_wake_coords --> _006G, _006H, _006I, _006J
# SHAPES: (1, 4, 3, 3) --> (1, 3, 2, 3), (1, 3, 2, 3), (1, 3, 2, 3), (1, 3, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v256__006G = ((v255_wing_TE_wake_coords.flatten())[src_indices__006G__006F]).reshape((1, 3, 2, 3))
v257__006H = ((v255_wing_TE_wake_coords.flatten())[src_indices__006H__006F]).reshape((1, 3, 2, 3))
v258__006I = ((v255_wing_TE_wake_coords.flatten())[src_indices__006I__006F]).reshape((1, 3, 2, 3))
v259__006J = ((v255_wing_TE_wake_coords.flatten())[src_indices__006J__006F]).reshape((1, 3, 2, 3))

# op _002M_linear_combination_eval
# LANG: _002I, _002L --> _002N
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v113__002N = v110__002I+v112__002L

# op _0035_linear_combination_eval
# LANG: _002J, _0034 --> _0036
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v124__0036 = v111__002J+v123__0034

# op _006K reshape_eval
# LANG: wing_coll_pts_coords --> _006L
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v260__006L = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _006Q reshape_eval
# LANG: _006G --> _006R
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v263__006R = v256__006G.reshape((1, 6, 3))

# op _0073 reshape_eval
# LANG: wing_coll_pts_coords --> _0074
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v270__0074 = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _0079 reshape_eval
# LANG: _006H --> _007a
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v273__007a = v257__006H.reshape((1, 6, 3))

# op _007n reshape_eval
# LANG: wing_coll_pts_coords --> _007o
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v280__007o = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007t reshape_eval
# LANG: _006I --> _007u
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v283__007u = v258__006I.reshape((1, 6, 3))

# op _002O_indexed_passthrough_eval
# LANG: _002N, _0036 --> wing_bd_vtx_coords
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 2, 3, 3)
# full namespace: MeshPreprocessing_comp
v434_wing_bd_vtx_coords__temp[i_v113__002N__002O_indexed_passthrough_eval] = v113__002N.flatten()
v434_wing_bd_vtx_coords = v434_wing_bd_vtx_coords__temp.copy()
v434_wing_bd_vtx_coords__temp[i_v124__0036__002O_indexed_passthrough_eval] = v124__0036.flatten()
v434_wing_bd_vtx_coords = v434_wing_bd_vtx_coords__temp.copy()

# op _006M expand_array_eval
# LANG: _006L --> _006N
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v261__006N = np.einsum('abd,c->abcd', v260__006L.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _006S expand_array_eval
# LANG: _006R --> _006T
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v264__006T = np.einsum('acd,b->abcd', v263__006R.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _0075 expand_array_eval
# LANG: _0074 --> _0076
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v271__0076 = np.einsum('abd,c->abcd', v270__0074.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007H reshape_eval
# LANG: wing_coll_pts_coords --> _007I
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v290__007I = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _007N reshape_eval
# LANG: _006J --> _007O
# SHAPES: (1, 3, 2, 3) --> (1, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v293__007O = v259__006J.reshape((1, 6, 3))

# op _007b expand_array_eval
# LANG: _007a --> _007c
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v274__007c = np.einsum('acd,b->abcd', v273__007a.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _007p expand_array_eval
# LANG: _007o --> _007q
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v281__007q = np.einsum('abd,c->abcd', v280__007o.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007v expand_array_eval
# LANG: _007u --> _007w
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v284__007w = np.einsum('acd,b->abcd', v283__007u.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _006O reshape_eval
# LANG: _006N --> _006P
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v262__006P = v261__006N.reshape((1, 12, 3))

# op _006U reshape_eval
# LANG: _006T --> _006V
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v265__006V = v264__006T.reshape((1, 12, 3))

# op _0077 reshape_eval
# LANG: _0076 --> _0078
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v272__0078 = v271__0076.reshape((1, 12, 3))

# op _007J expand_array_eval
# LANG: _007I --> _007K
# SHAPES: (1, 2, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v291__007K = np.einsum('abd,c->abcd', v290__007I.reshape((1, 2, 3)) ,np.ones((6,))).reshape((1, 2, 6, 3))

# op _007P expand_array_eval
# LANG: _007O --> _007Q
# SHAPES: (1, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v294__007Q = np.einsum('acd,b->abcd', v293__007O.reshape((1, 6, 3)) ,np.ones((2,))).reshape((1, 2, 6, 3))

# op _007d reshape_eval
# LANG: _007c --> _007e
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v275__007e = v274__007c.reshape((1, 12, 3))

# op _007r reshape_eval
# LANG: _007q --> _007s
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v282__007s = v281__007q.reshape((1, 12, 3))

# op _007x reshape_eval
# LANG: _007w --> _007y
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v285__007y = v284__007w.reshape((1, 12, 3))

# op _00cc_decompose_eval
# LANG: wing_bd_vtx_coords --> _00cd, _00ce, _00cf, _00cg
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v435__00cd = ((v434_wing_bd_vtx_coords.flatten())[src_indices__00cd__00cc]).reshape((1, 1, 2, 3))
v436__00ce = ((v434_wing_bd_vtx_coords.flatten())[src_indices__00ce__00cc]).reshape((1, 1, 2, 3))
v437__00cf = ((v434_wing_bd_vtx_coords.flatten())[src_indices__00cf__00cc]).reshape((1, 1, 2, 3))
v438__00cg = ((v434_wing_bd_vtx_coords.flatten())[src_indices__00cg__00cc]).reshape((1, 1, 2, 3))

# op _006W_linear_combination_eval
# LANG: _006P, _006V --> _006X
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v266__006X = v262__006P+-1*v265__006V

# op _007L reshape_eval
# LANG: _007K --> _007M
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v292__007M = v291__007K.reshape((1, 12, 3))

# op _007R reshape_eval
# LANG: _007Q --> _007S
# SHAPES: (1, 2, 6, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v295__007S = v294__007Q.reshape((1, 12, 3))

# op _007f_linear_combination_eval
# LANG: _0078, _007e --> _007g
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v276__007g = v272__0078+-1*v275__007e

# op _007z_linear_combination_eval
# LANG: _007s, _007y --> _007A
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v286__007A = v282__007s+-1*v285__007y

# op _00cB reshape_eval
# LANG: wing_coll_pts_coords --> _00cC
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v449__00cC = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00cH reshape_eval
# LANG: _00ce --> _00cI
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v452__00cI = v436__00ce.reshape((1, 2, 3))

# op _00cV reshape_eval
# LANG: wing_coll_pts_coords --> _00cW
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v459__00cW = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00ch reshape_eval
# LANG: wing_coll_pts_coords --> _00ci
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v439__00ci = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00cn reshape_eval
# LANG: _00cd --> _00co
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v442__00co = v435__00cd.reshape((1, 2, 3))

# op _00d0 reshape_eval
# LANG: _00cf --> _00d1
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v462__00d1 = v437__00cf.reshape((1, 2, 3))

# op _006Y_power_combination_eval
# LANG: _006X --> _006Z
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v267__006Z = (v266__006X**2)
v267__006Z = v267__006Z.reshape((1, 12, 3))

# op _007B_power_combination_eval
# LANG: _007A --> _007C
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v287__007C = (v286__007A**2)
v287__007C = v287__007C.reshape((1, 12, 3))

# op _007T_linear_combination_eval
# LANG: _007M, _007S --> _007U
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v296__007U = v292__007M+-1*v295__007S

# op _007h_power_combination_eval
# LANG: _007g --> _007i
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v277__007i = (v276__007g**2)
v277__007i = v277__007i.reshape((1, 12, 3))

# op _00cD expand_array_eval
# LANG: _00cC --> _00cE
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v450__00cE = np.einsum('abd,c->abcd', v449__00cC.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cJ expand_array_eval
# LANG: _00cI --> _00cK
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v453__00cK = np.einsum('acd,b->abcd', v452__00cI.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cX expand_array_eval
# LANG: _00cW --> _00cY
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v460__00cY = np.einsum('abd,c->abcd', v459__00cW.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cj expand_array_eval
# LANG: _00ci --> _00ck
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v440__00ck = np.einsum('abd,c->abcd', v439__00ci.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00cp expand_array_eval
# LANG: _00co --> _00cq
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v443__00cq = np.einsum('acd,b->abcd', v442__00co.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00d2 expand_array_eval
# LANG: _00d1 --> _00d3
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v463__00d3 = np.einsum('acd,b->abcd', v462__00d1.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00de reshape_eval
# LANG: wing_coll_pts_coords --> _00df
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v469__00df = v433_wing_coll_pts_coords.reshape((1, 2, 3))

# op _00dk reshape_eval
# LANG: _00cg --> _00dl
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v472__00dl = v438__00cg.reshape((1, 2, 3))

# op _006__single_tensor_sum_with_axis_eval
# LANG: _006Z --> _0070
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v268__0070 = np.sum(v267__006Z, axis = (2,)).reshape((1, 12))

# op _007D_single_tensor_sum_with_axis_eval
# LANG: _007C --> _007E
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v288__007E = np.sum(v287__007C, axis = (2,)).reshape((1, 12))

# op _007V_power_combination_eval
# LANG: _007U --> _007W
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v297__007W = (v296__007U**2)
v297__007W = v297__007W.reshape((1, 12, 3))

# op _007j_single_tensor_sum_with_axis_eval
# LANG: _007i --> _007k
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v278__007k = np.sum(v277__007i, axis = (2,)).reshape((1, 12))

# op _00cF reshape_eval
# LANG: _00cE --> _00cG
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v451__00cG = v450__00cE.reshape((1, 4, 3))

# op _00cL reshape_eval
# LANG: _00cK --> _00cM
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v454__00cM = v453__00cK.reshape((1, 4, 3))

# op _00cZ reshape_eval
# LANG: _00cY --> _00c_
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v461__00c_ = v460__00cY.reshape((1, 4, 3))

# op _00cl reshape_eval
# LANG: _00ck --> _00cm
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v441__00cm = v440__00ck.reshape((1, 4, 3))

# op _00cr reshape_eval
# LANG: _00cq --> _00cs
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v444__00cs = v443__00cq.reshape((1, 4, 3))

# op _00d4 reshape_eval
# LANG: _00d3 --> _00d5
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v464__00d5 = v463__00d3.reshape((1, 4, 3))

# op _00dg expand_array_eval
# LANG: _00df --> _00dh
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v470__00dh = np.einsum('abd,c->abcd', v469__00df.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _00dm expand_array_eval
# LANG: _00dl --> _00dn
# SHAPES: (1, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v473__00dn = np.einsum('acd,b->abcd', v472__00dl.reshape((1, 2, 3)) ,np.ones((2,))).reshape((1, 2, 2, 3))

# op _0071_power_combination_eval
# LANG: _0070 --> _0072
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v269__0072 = (v268__0070**0.5)
v269__0072 = v269__0072.reshape((1, 12))

# op _007F_power_combination_eval
# LANG: _007E --> _007G
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v289__007G = (v288__007E**0.5)
v289__007G = v289__007G.reshape((1, 12))

# op _007X_single_tensor_sum_with_axis_eval
# LANG: _007W --> _007Y
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v298__007Y = np.sum(v297__007W, axis = (2,)).reshape((1, 12))

# op _007l_power_combination_eval
# LANG: _007k --> _007m
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v279__007m = (v278__007k**0.5)
v279__007m = v279__007m.reshape((1, 12))

# op _00cN_linear_combination_eval
# LANG: _00cG, _00cM --> _00cO
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v455__00cO = v451__00cG+-1*v454__00cM

# op _00ct_linear_combination_eval
# LANG: _00cm, _00cs --> _00cu
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v445__00cu = v441__00cm+-1*v444__00cs

# op _00d6_linear_combination_eval
# LANG: _00c_, _00d5 --> _00d7
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v465__00d7 = v461__00c_+-1*v464__00d5

# op _00di reshape_eval
# LANG: _00dh --> _00dj
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v471__00dj = v470__00dh.reshape((1, 4, 3))

# op _00do reshape_eval
# LANG: _00dn --> _00dp
# SHAPES: (1, 2, 2, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v474__00dp = v473__00dn.reshape((1, 4, 3))

# op _007Z_power_combination_eval
# LANG: _007Y --> _007_
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v299__007_ = (v298__007Y**0.5)
v299__007_ = v299__007_.reshape((1, 12))

# op _0084_power_combination_eval
# LANG: _006X, _007g --> _0085
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v302__0085 = (v266__006X)*(v276__007g)
v302__0085 = v302__0085.reshape((1, 12, 3))

# op _0088_power_combination_eval
# LANG: _0072 --> _0089
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v304__0089 = (v269__0072**2)
v304__0089 = v304__0089.reshape((1, 12))

# op _008G_power_combination_eval
# LANG: _0072 --> _008H
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v321__008H = (v269__0072)
v321__008H = (v321__008H*_008G_coeff).reshape((1, 12))

# op _008a_power_combination_eval
# LANG: _007m --> _008b
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v305__008b = (v279__007m**2)
v305__008b = v305__008b.reshape((1, 12))

# op _0091_power_combination_eval
# LANG: _007A, _007g --> _0092
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v332__0092 = (v276__007g)*(v286__007A)
v332__0092 = v332__0092.reshape((1, 12, 3))

# op _0095_power_combination_eval
# LANG: _007m --> _0096
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v334__0096 = (v279__007m**2)
v334__0096 = v334__0096.reshape((1, 12))

# op _0097_power_combination_eval
# LANG: _007G --> _0098
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v335__0098 = (v289__007G**2)
v335__0098 = v335__0098.reshape((1, 12))

# op _009D_power_combination_eval
# LANG: _007m --> _009E
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v351__009E = (v279__007m)
v351__009E = (v351__009E*_009D_coeff).reshape((1, 12))

# op _00cP_power_combination_eval
# LANG: _00cO --> _00cQ
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v456__00cQ = (v455__00cO**2)
v456__00cQ = v456__00cQ.reshape((1, 4, 3))

# op _00cv_power_combination_eval
# LANG: _00cu --> _00cw
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v446__00cw = (v445__00cu**2)
v446__00cw = v446__00cw.reshape((1, 4, 3))

# op _00d8_power_combination_eval
# LANG: _00d7 --> _00d9
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v466__00d9 = (v465__00d7**2)
v466__00d9 = v466__00d9.reshape((1, 4, 3))

# op _00dq_linear_combination_eval
# LANG: _00dj, _00dp --> _00dr
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v475__00dr = v471__00dj+-1*v474__00dp

# op _0086_single_tensor_sum_with_axis_eval
# LANG: _0085 --> _0087
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v303__0087 = np.sum(v302__0085, axis = (2,)).reshape((1, 12))

# op _008E_linear_combination_eval
# LANG: _0089, _008b --> _008F
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v320__008F = v304__0089+v305__008b

# op _008I_power_combination_eval
# LANG: _007m, _008H --> _008J
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v322__008J = (v321__008H)*(v279__007m)
v322__008J = v322__008J.reshape((1, 12))

# op _008e_linear_combination_eval
# LANG: _0089 --> _008f
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v307__008f = _008e_constant+v304__0089

# op _008o_linear_combination_eval
# LANG: _008b --> _008p
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v312__008p = _008o_constant+v305__008b

# op _0093_single_tensor_sum_with_axis_eval
# LANG: _0092 --> _0094
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v333__0094 = np.sum(v332__0092, axis = (2,)).reshape((1, 12))

# op _009B_linear_combination_eval
# LANG: _0096, _0098 --> _009C
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v350__009C = v334__0096+v335__0098

# op _009F_power_combination_eval
# LANG: _007G, _009E --> _009G
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v352__009G = (v351__009E)*(v289__007G)
v352__009G = v352__009G.reshape((1, 12))

# op _009Z_power_combination_eval
# LANG: _007U, _007A --> _009_
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v362__009_ = (v286__007A)*(v296__007U)
v362__009_ = v362__009_.reshape((1, 12, 3))

# op _009b_linear_combination_eval
# LANG: _0096 --> _009c
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v337__009c = _009b_constant+v334__0096

# op _009l_linear_combination_eval
# LANG: _0098 --> _009m
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v342__009m = _009l_constant+v335__0098

# op _00a2_power_combination_eval
# LANG: _007G --> _00a3
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v364__00a3 = (v289__007G**2)
v364__00a3 = v364__00a3.reshape((1, 12))

# op _00a4_power_combination_eval
# LANG: _007_ --> _00a5
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v365__00a5 = (v299__007_**2)
v365__00a5 = v365__00a5.reshape((1, 12))

# op _00aA_power_combination_eval
# LANG: _007G --> _00aB
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v381__00aB = (v289__007G)
v381__00aB = (v381__00aB*_00aA_coeff).reshape((1, 12))

# op _00cR_single_tensor_sum_with_axis_eval
# LANG: _00cQ --> _00cS
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v457__00cS = np.sum(v456__00cQ, axis = (2,)).reshape((1, 4))

# op _00cx_single_tensor_sum_with_axis_eval
# LANG: _00cw --> _00cy
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v447__00cy = np.sum(v446__00cw, axis = (2,)).reshape((1, 4))

# op _00da_single_tensor_sum_with_axis_eval
# LANG: _00d9 --> _00db
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v467__00db = np.sum(v466__00d9, axis = (2,)).reshape((1, 4))

# op _00ds_power_combination_eval
# LANG: _00dr --> _00dt
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v476__00dt = (v475__00dr**2)
v476__00dt = v476__00dt.reshape((1, 4, 3))

# op _008A_power_combination_eval
# LANG: _0087 --> _008B
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v318__008B = (v303__0087**2)
v318__008B = v318__008B.reshape((1, 12))

# op _008K_linear_combination_eval
# LANG: _008F, _008J --> _008L
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v323__008L = v320__008F+-1*v322__008J

# op _008g_linear_combination_eval
# LANG: _008f --> _008h
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v308__008h = _008g_constant+v307__008f

# op _008q_linear_combination_eval
# LANG: _008p --> _008r
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v313__008r = _008q_constant+v312__008p

# op _008y_power_combination_eval
# LANG: _0089, _008b --> _008z
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v317__008z = (v304__0089)*(v305__008b)
v317__008z = v317__008z.reshape((1, 12))

# op _009H_linear_combination_eval
# LANG: _009C, _009G --> _009I
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v353__009I = v350__009C+-1*v352__009G

# op _009d_linear_combination_eval
# LANG: _009c --> _009e
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v338__009e = _009d_constant+v337__009c

# op _009n_linear_combination_eval
# LANG: _009m --> _009o
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v343__009o = _009n_constant+v342__009m

# op _009v_power_combination_eval
# LANG: _0096, _0098 --> _009w
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v347__009w = (v334__0096)*(v335__0098)
v347__009w = v347__009w.reshape((1, 12))

# op _009x_power_combination_eval
# LANG: _0094 --> _009y
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v348__009y = (v333__0094**2)
v348__009y = v348__009y.reshape((1, 12))

# op _00a0_single_tensor_sum_with_axis_eval
# LANG: _009_ --> _00a1
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v363__00a1 = np.sum(v362__009_, axis = (2,)).reshape((1, 12))

# op _00a8_linear_combination_eval
# LANG: _00a3 --> _00a9
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v367__00a9 = _00a8_constant+v364__00a3

# op _00aC_power_combination_eval
# LANG: _007_, _00aB --> _00aD
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v382__00aD = (v381__00aB)*(v299__007_)
v382__00aD = v382__00aD.reshape((1, 12))

# op _00aW_power_combination_eval
# LANG: _007U, _006X --> _00aX
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v392__00aX = (v296__007U)*(v266__006X)
v392__00aX = v392__00aX.reshape((1, 12, 3))

# op _00a__power_combination_eval
# LANG: _007_ --> _00b0
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v394__00b0 = (v299__007_**2)
v394__00b0 = v394__00b0.reshape((1, 12))

# op _00ai_linear_combination_eval
# LANG: _00a5 --> _00aj
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v372__00aj = _00ai_constant+v365__00a5

# op _00ay_linear_combination_eval
# LANG: _00a3, _00a5 --> _00az
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v380__00az = v364__00a3+v365__00a5

# op _00b1_power_combination_eval
# LANG: _0072 --> _00b2
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v395__00b2 = (v269__0072**2)
v395__00b2 = v395__00b2.reshape((1, 12))

# op _00bx_power_combination_eval
# LANG: _007_ --> _00by
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v411__00by = (v299__007_)
v411__00by = (v411__00by*_00bx_coeff).reshape((1, 12))

# op _00cT_power_combination_eval
# LANG: _00cS --> _00cU
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v458__00cU = (v457__00cS**0.5)
v458__00cU = v458__00cU.reshape((1, 4))

# op _00cz_power_combination_eval
# LANG: _00cy --> _00cA
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v448__00cA = (v447__00cy**0.5)
v448__00cA = v448__00cA.reshape((1, 4))

# op _00dC_power_combination_eval
# LANG: _00cu, _00cO --> _00dD
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v481__00dD = (v445__00cu)*(v455__00cO)
v481__00dD = v481__00dD.reshape((1, 4, 3))

# op _00dc_power_combination_eval
# LANG: _00db --> _00dd
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v468__00dd = (v467__00db**0.5)
v468__00dd = v468__00dd.reshape((1, 4))

# op _00du_single_tensor_sum_with_axis_eval
# LANG: _00dt --> _00dv
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v477__00dv = np.sum(v476__00dt, axis = (2,)).reshape((1, 4))

# op _00e1_power_combination_eval
# LANG: _00d7, _00cO --> _00e2
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v494__00e2 = (v455__00cO)*(v465__00d7)
v494__00e2 = v494__00e2.reshape((1, 4, 3))

# op _008C_linear_combination_eval
# LANG: _008z, _008B --> _008D
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v319__008D = v317__008z+-1*v318__008B

# op _008M_power_combination_eval
# LANG: _008L --> _008N
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v324__008N = (v323__008L)
v324__008N = (v324__008N*_008M_coeff).reshape((1, 12))

# op _008c_linear_combination_eval
# LANG: _0089, _0087 --> _008d
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v306__008d = v304__0089+-1*v303__0087

# op _008i_power_combination_eval
# LANG: _008h --> _008j
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v309__008j = (v308__008h**0.5)
v309__008j = v309__008j.reshape((1, 12))

# op _008m_linear_combination_eval
# LANG: _008b, _0087 --> _008n
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v311__008n = v305__008b+-1*v303__0087

# op _008s_power_combination_eval
# LANG: _008r --> _008t
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v314__008t = (v313__008r**0.5)
v314__008t = v314__008t.reshape((1, 12))

# op _0099_linear_combination_eval
# LANG: _0096, _0094 --> _009a
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v336__009a = v334__0096+-1*v333__0094

# op _009J_power_combination_eval
# LANG: _009I --> _009K
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v354__009K = (v353__009I)
v354__009K = (v354__009K*_009J_coeff).reshape((1, 12))

# op _009f_power_combination_eval
# LANG: _009e --> _009g
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v339__009g = (v338__009e**0.5)
v339__009g = v339__009g.reshape((1, 12))

# op _009j_linear_combination_eval
# LANG: _0098, _0094 --> _009k
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v341__009k = v335__0098+-1*v333__0094

# op _009p_power_combination_eval
# LANG: _009o --> _009q
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v344__009q = (v343__009o**0.5)
v344__009q = v344__009q.reshape((1, 12))

# op _009z_linear_combination_eval
# LANG: _009w, _009y --> _009A
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v349__009A = v347__009w+-1*v348__009y

# op _00aE_linear_combination_eval
# LANG: _00az, _00aD --> _00aF
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v383__00aF = v380__00az+-1*v382__00aD

# op _00aY_single_tensor_sum_with_axis_eval
# LANG: _00aX --> _00aZ
# SHAPES: (1, 12, 3) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v393__00aZ = np.sum(v392__00aX, axis = (2,)).reshape((1, 12))

# op _00aa_linear_combination_eval
# LANG: _00a9 --> _00ab
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v368__00ab = _00aa_constant+v367__00a9

# op _00ak_linear_combination_eval
# LANG: _00aj --> _00al
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v373__00al = _00ak_constant+v372__00aj

# op _00as_power_combination_eval
# LANG: _00a3, _00a5 --> _00at
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v377__00at = (v364__00a3)*(v365__00a5)
v377__00at = v377__00at.reshape((1, 12))

# op _00au_power_combination_eval
# LANG: _00a1 --> _00av
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v378__00av = (v363__00a1**2)
v378__00av = v378__00av.reshape((1, 12))

# op _00b5_linear_combination_eval
# LANG: _00b0 --> _00b6
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v397__00b6 = _00b5_constant+v394__00b0

# op _00bf_linear_combination_eval
# LANG: _00b2 --> _00bg
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v402__00bg = _00bf_constant+v395__00b2

# op _00bv_linear_combination_eval
# LANG: _00b0, _00b2 --> _00bw
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v410__00bw = v394__00b0+v395__00b2

# op _00bz_power_combination_eval
# LANG: _00by, _0072 --> _00bA
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v412__00bA = (v411__00by)*(v269__0072)
v412__00bA = v412__00bA.reshape((1, 12))

# op _00dE_single_tensor_sum_with_axis_eval
# LANG: _00dD --> _00dF
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v482__00dF = np.sum(v481__00dD, axis = (2,)).reshape((1, 4))

# op _00dG_power_combination_eval
# LANG: _00cA, _00cU --> _00dH
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v483__00dH = (v448__00cA)*(v458__00cU)
v483__00dH = v483__00dH.reshape((1, 4))

# op _00dw_power_combination_eval
# LANG: _00dv --> _00dx
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v478__00dx = (v477__00dv**0.5)
v478__00dx = v478__00dx.reshape((1, 4))

# op _00e3_single_tensor_sum_with_axis_eval
# LANG: _00e2 --> _00e4
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v495__00e4 = np.sum(v494__00e2, axis = (2,)).reshape((1, 4))

# op _00e5_power_combination_eval
# LANG: _00dd, _00cU --> _00e6
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v496__00e6 = (v458__00cU)*(v468__00dd)
v496__00e6 = v496__00e6.reshape((1, 4))

# op _00er_power_combination_eval
# LANG: _00dr, _00d7 --> _00es
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v507__00es = (v465__00d7)*(v475__00dr)
v507__00es = v507__00es.reshape((1, 4, 3))

# op _0062_decompose_eval
# LANG: wing --> _0068, _0063, _0064, _0067
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v232__0063 = ((v231_wing.flatten())[src_indices__0063__0062]).reshape((1, 1, 2, 3))
v233__0064 = ((v231_wing.flatten())[src_indices__0064__0062]).reshape((1, 1, 2, 3))
v235__0067 = ((v231_wing.flatten())[src_indices__0067__0062]).reshape((1, 1, 2, 3))
v236__0068 = ((v231_wing.flatten())[src_indices__0068__0062]).reshape((1, 1, 2, 3))

# op _008O_linear_combination_eval
# LANG: _008D, _008N --> _008P
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v325__008P = v319__008D+v324__008N

# op _008k_power_combination_eval
# LANG: _008d, _008j --> _008l
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v310__008l = (v306__008d)*(v309__008j**-1)
v310__008l = v310__008l.reshape((1, 12))

# op _008u_power_combination_eval
# LANG: _008n, _008t --> _008v
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v315__008v = (v311__008n)*(v314__008t**-1)
v315__008v = v315__008v.reshape((1, 12))

# op _009L_linear_combination_eval
# LANG: _009A, _009K --> _009M
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v355__009M = v349__009A+v354__009K

# op _009h_power_combination_eval
# LANG: _009a, _009g --> _009i
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v340__009i = (v336__009a)*(v339__009g**-1)
v340__009i = v340__009i.reshape((1, 12))

# op _009r_power_combination_eval
# LANG: _009k, _009q --> _009s
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v345__009s = (v341__009k)*(v344__009q**-1)
v345__009s = v345__009s.reshape((1, 12))

# op _00a6_linear_combination_eval
# LANG: _00a3, _00a1 --> _00a7
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v366__00a7 = v364__00a3+-1*v363__00a1

# op _00aG_power_combination_eval
# LANG: _00aF --> _00aH
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v384__00aH = (v383__00aF)
v384__00aH = (v384__00aH*_00aG_coeff).reshape((1, 12))

# op _00ac_power_combination_eval
# LANG: _00ab --> _00ad
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v369__00ad = (v368__00ab**0.5)
v369__00ad = v369__00ad.reshape((1, 12))

# op _00ag_linear_combination_eval
# LANG: _00a5, _00a1 --> _00ah
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v371__00ah = v365__00a5+-1*v363__00a1

# op _00am_power_combination_eval
# LANG: _00al --> _00an
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v374__00an = (v373__00al**0.5)
v374__00an = v374__00an.reshape((1, 12))

# op _00aw_linear_combination_eval
# LANG: _00at, _00av --> _00ax
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v379__00ax = v377__00at+-1*v378__00av

# op _00b7_linear_combination_eval
# LANG: _00b6 --> _00b8
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v398__00b8 = _00b7_constant+v397__00b6

# op _00bB_linear_combination_eval
# LANG: _00bw, _00bA --> _00bC
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v413__00bC = v410__00bw+-1*v412__00bA

# op _00bh_linear_combination_eval
# LANG: _00bg --> _00bi
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v403__00bi = _00bh_constant+v402__00bg

# op _00bp_power_combination_eval
# LANG: _00b0, _00b2 --> _00bq
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v407__00bq = (v394__00b0)*(v395__00b2)
v407__00bq = v407__00bq.reshape((1, 12))

# op _00br_power_combination_eval
# LANG: _00aZ --> _00bs
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v408__00bs = (v393__00aZ**2)
v408__00bs = v408__00bs.reshape((1, 12))

# op _00dI_linear_combination_eval
# LANG: _00dH, _00dF --> _00dJ
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v484__00dJ = v483__00dH+v482__00dF

# op _00dM_power_combination_eval
# LANG: _00cA --> _00dN
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v486__00dN = (v448__00cA**-1)
v486__00dN = v486__00dN.reshape((1, 4))

# op _00dO_power_combination_eval
# LANG: _00cU --> _00dP
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v487__00dP = (v458__00cU**-1)
v487__00dP = v487__00dP.reshape((1, 4))

# op _00e7_linear_combination_eval
# LANG: _00e6, _00e4 --> _00e8
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v497__00e8 = v496__00e6+v495__00e4

# op _00eR_power_combination_eval
# LANG: _00dr, _00cu --> _00eS
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v520__00eS = (v475__00dr)*(v445__00cu)
v520__00eS = v520__00eS.reshape((1, 4, 3))

# op _00eb_power_combination_eval
# LANG: _00cU --> _00ec
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v499__00ec = (v458__00cU**-1)
v499__00ec = v499__00ec.reshape((1, 4))

# op _00ed_power_combination_eval
# LANG: _00dd --> _00ee
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v500__00ee = (v468__00dd**-1)
v500__00ee = v500__00ee.reshape((1, 4))

# op _00et_single_tensor_sum_with_axis_eval
# LANG: _00es --> _00eu
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v508__00eu = np.sum(v507__00es, axis = (2,)).reshape((1, 4))

# op _00ev_power_combination_eval
# LANG: _00dx, _00dd --> _00ew
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v509__00ew = (v468__00dd)*(v478__00dx)
v509__00ew = v509__00ew.reshape((1, 4))

# op _0065_linear_combination_eval
# LANG: _0063, _0064 --> _0066
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v234__0066 = v232__0063+-1*v233__0064

# op _0069_linear_combination_eval
# LANG: _0067, _0068 --> _006a
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v237__006a = v235__0067+-1*v236__0068

# op _008Q_linear_combination_eval
# LANG: _008P --> _008R
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v326__008R = _008Q_constant+v325__008P

# op _008w_linear_combination_eval
# LANG: _008l, _008v --> _008x
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v316__008x = v310__008l+v315__008v

# op _009N_linear_combination_eval
# LANG: _009M --> _009O
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v356__009O = _009N_constant+v355__009M

# op _009t_linear_combination_eval
# LANG: _009i, _009s --> _009u
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v346__009u = v340__009i+v345__009s

# op _00aI_linear_combination_eval
# LANG: _00ax, _00aH --> _00aJ
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v385__00aJ = v379__00ax+v384__00aH

# op _00ae_power_combination_eval
# LANG: _00a7, _00ad --> _00af
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v370__00af = (v366__00a7)*(v369__00ad**-1)
v370__00af = v370__00af.reshape((1, 12))

# op _00ao_power_combination_eval
# LANG: _00ah, _00an --> _00ap
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v375__00ap = (v371__00ah)*(v374__00an**-1)
v375__00ap = v375__00ap.reshape((1, 12))

# op _00b3_linear_combination_eval
# LANG: _00b0, _00aZ --> _00b4
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v396__00b4 = v394__00b0+-1*v393__00aZ

# op _00b9_power_combination_eval
# LANG: _00b8 --> _00ba
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v399__00ba = (v398__00b8**0.5)
v399__00ba = v399__00ba.reshape((1, 12))

# op _00bD_power_combination_eval
# LANG: _00bC --> _00bE
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v414__00bE = (v413__00bC)
v414__00bE = (v414__00bE*_00bD_coeff).reshape((1, 12))

# op _00bd_linear_combination_eval
# LANG: _00b2, _00aZ --> _00be
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v401__00be = v395__00b2+-1*v393__00aZ

# op _00bj_power_combination_eval
# LANG: _00bi --> _00bk
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v404__00bk = (v403__00bi**0.5)
v404__00bk = v404__00bk.reshape((1, 12))

# op _00bt_linear_combination_eval
# LANG: _00bq, _00bs --> _00bu
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v409__00bu = v407__00bq+-1*v408__00bs

# op _00dK_power_combination_eval
# LANG: _00dJ --> _00dL
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v485__00dL = (v484__00dJ**-1)
v485__00dL = v485__00dL.reshape((1, 4))

# op _00dQ_linear_combination_eval
# LANG: _00dN, _00dP --> _00dR
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v488__00dR = v486__00dN+v487__00dP

# op _00e9_power_combination_eval
# LANG: _00e8 --> _00ea
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v498__00ea = (v497__00e8**-1)
v498__00ea = v498__00ea.reshape((1, 4))

# op _00eB_power_combination_eval
# LANG: _00dd --> _00eC
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v512__00eC = (v468__00dd**-1)
v512__00eC = v512__00eC.reshape((1, 4))

# op _00eD_power_combination_eval
# LANG: _00dx --> _00eE
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v513__00eE = (v478__00dx**-1)
v513__00eE = v513__00eE.reshape((1, 4))

# op _00eT_single_tensor_sum_with_axis_eval
# LANG: _00eS --> _00eU
# SHAPES: (1, 4, 3) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v521__00eU = np.sum(v520__00eS, axis = (2,)).reshape((1, 4))

# op _00eV_power_combination_eval
# LANG: _00cA, _00dx --> _00eW
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v522__00eW = (v478__00dx)*(v448__00cA)
v522__00eW = v522__00eW.reshape((1, 4))

# op _00ef_linear_combination_eval
# LANG: _00ec, _00ee --> _00eg
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v501__00eg = v499__00ec+v500__00ee

# op _00ex_linear_combination_eval
# LANG: _00ew, _00eu --> _00ey
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v510__00ey = v509__00ew+v508__00eu

# op _006b cross_product_eval
# LANG: _0066, _006a --> _006c
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v238__006c = np.cross(v234__0066, v237__006a, axisa = 3, axisb = 3, axisc = 3)

# op _0080 cross_product_eval
# LANG: _006X, _007g --> _0081
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v300__0081 = np.cross(v266__006X, v276__007g, axisa = 2, axisb = 2, axisc = 2)

# op _008S_power_combination_eval
# LANG: _008x, _008R --> _008T
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v327__008T = (v316__008x)*(v326__008R**-1)
v327__008T = v327__008T.reshape((1, 12))

# op _008Y cross_product_eval
# LANG: _007A, _007g --> _008Z
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v330__008Z = np.cross(v276__007g, v286__007A, axisa = 2, axisb = 2, axisc = 2)

# op _009P_power_combination_eval
# LANG: _009u, _009O --> _009Q
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v357__009Q = (v346__009u)*(v356__009O**-1)
v357__009Q = v357__009Q.reshape((1, 12))

# op _00aK_linear_combination_eval
# LANG: _00aJ --> _00aL
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v386__00aL = _00aK_constant+v385__00aJ

# op _00aq_linear_combination_eval
# LANG: _00af, _00ap --> _00ar
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v376__00ar = v370__00af+v375__00ap

# op _00bF_linear_combination_eval
# LANG: _00bu, _00bE --> _00bG
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v415__00bG = v409__00bu+v414__00bE

# op _00bb_power_combination_eval
# LANG: _00b4, _00ba --> _00bc
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v400__00bc = (v396__00b4)*(v399__00ba**-1)
v400__00bc = v400__00bc.reshape((1, 12))

# op _00bl_power_combination_eval
# LANG: _00be, _00bk --> _00bm
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v405__00bm = (v401__00be)*(v404__00bk**-1)
v405__00bm = v405__00bm.reshape((1, 12))

# op _00dS_power_combination_eval
# LANG: _00dL, _00dR --> _00dT
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v489__00dT = (v485__00dL)*(v488__00dR)
v489__00dT = v489__00dT.reshape((1, 4))

# op _00dY cross_product_eval
# LANG: _00d7, _00cO --> _00dZ
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v492__00dZ = np.cross(v455__00cO, v465__00d7, axisa = 2, axisb = 2, axisc = 2)

# op _00dy cross_product_eval
# LANG: _00cu, _00cO --> _00dz
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v479__00dz = np.cross(v445__00cu, v455__00cO, axisa = 2, axisb = 2, axisc = 2)

# op _00eF_linear_combination_eval
# LANG: _00eC, _00eE --> _00eG
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v514__00eG = v512__00eC+v513__00eE

# op _00eX_linear_combination_eval
# LANG: _00eW, _00eU --> _00eY
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v523__00eY = v522__00eW+v521__00eU

# op _00eh_power_combination_eval
# LANG: _00ea, _00eg --> _00ei
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v502__00ei = (v498__00ea)*(v501__00eg)
v502__00ei = v502__00ei.reshape((1, 4))

# op _00ez_power_combination_eval
# LANG: _00ey --> _00eA
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v511__00eA = (v510__00ey**-1)
v511__00eA = v511__00eA.reshape((1, 4))

# op _00f0_power_combination_eval
# LANG: _00dx --> _00f1
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v525__00f1 = (v478__00dx**-1)
v525__00f1 = v525__00f1.reshape((1, 4))

# op _00f2_power_combination_eval
# LANG: _00cA --> _00f3
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v526__00f3 = (v448__00cA**-1)
v526__00f3 = v526__00f3.reshape((1, 4))

# op _005E_indexed_passthrough_eval
# LANG: p, q, r --> ang_vel
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v217_ang_vel__temp[i_v214_p__005E_indexed_passthrough_eval] = v214_p.flatten()
v217_ang_vel = v217_ang_vel__temp.copy()
v217_ang_vel__temp[i_v215_q__005E_indexed_passthrough_eval] = v215_q.flatten()
v217_ang_vel = v217_ang_vel__temp.copy()
v217_ang_vel__temp[i_v216_r__005E_indexed_passthrough_eval] = v216_r.flatten()
v217_ang_vel = v217_ang_vel__temp.copy()

# op _005H expand_array_eval
# LANG: wing_rot_ref --> _005I
# SHAPES: (1, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v220__005I = np.einsum('ad,bc->abcd', v219_wing_rot_ref.reshape((1, 3)) ,np.ones((1, 2))).reshape((1, 1, 2, 3))

# op _006d_power_combination_eval
# LANG: _006c --> _006e
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v239__006e = (v238__006c**2)
v239__006e = v239__006e.reshape((1, 1, 2, 3))

# op _0082_power_combination_eval
# LANG: _0081 --> _0083
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v301__0083 = (v300__0081)
v301__0083 = (v301__0083*_0082_coeff).reshape((1, 12, 3))

# op _008U expand_array_eval
# LANG: _008T --> _008V
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v328__008V = np.einsum('ab,c->abc', v327__008T.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _008__power_combination_eval
# LANG: _008Z --> _0090
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v331__0090 = (v330__008Z)
v331__0090 = (v331__0090*_008__coeff).reshape((1, 12, 3))

# op _009R expand_array_eval
# LANG: _009Q --> _009S
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v358__009S = np.einsum('ab,c->abc', v357__009Q.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _009V cross_product_eval
# LANG: _007U, _007A --> _009W
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v360__009W = np.cross(v286__007A, v296__007U, axisa = 2, axisb = 2, axisc = 2)

# op _00aM_power_combination_eval
# LANG: _00ar, _00aL --> _00aN
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v387__00aN = (v376__00ar)*(v386__00aL**-1)
v387__00aN = v387__00aN.reshape((1, 12))

# op _00bH_linear_combination_eval
# LANG: _00bG --> _00bI
# SHAPES: (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v416__00bI = _00bH_constant+v415__00bG

# op _00bn_linear_combination_eval
# LANG: _00bc, _00bm --> _00bo
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v406__00bo = v400__00bc+v405__00bm

# op _00dA_power_combination_eval
# LANG: _00dz --> _00dB
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v480__00dB = (v479__00dz)
v480__00dB = (v480__00dB*_00dA_coeff).reshape((1, 4, 3))

# op _00dU expand_array_eval
# LANG: _00dT --> _00dV
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v490__00dV = np.einsum('ab,c->abc', v489__00dT.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00d__power_combination_eval
# LANG: _00dZ --> _00e0
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v493__00e0 = (v492__00dZ)
v493__00e0 = (v493__00e0*_00d__coeff).reshape((1, 4, 3))

# op _00eH_power_combination_eval
# LANG: _00eA, _00eG --> _00eI
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v515__00eI = (v511__00eA)*(v514__00eG)
v515__00eI = v515__00eI.reshape((1, 4))

# op _00eZ_power_combination_eval
# LANG: _00eY --> _00e_
# SHAPES: (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v524__00e_ = (v523__00eY**-1)
v524__00e_ = v524__00e_.reshape((1, 4))

# op _00ej expand_array_eval
# LANG: _00ei --> _00ek
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v503__00ek = np.einsum('ab,c->abc', v502__00ei.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00en cross_product_eval
# LANG: _00dr, _00d7 --> _00eo
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v505__00eo = np.cross(v465__00d7, v475__00dr, axisa = 2, axisb = 2, axisc = 2)

# op _00f4_linear_combination_eval
# LANG: _00f1, _00f3 --> _00f5
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v527__00f5 = v525__00f1+v526__00f3

# op _005J_linear_combination_eval
# LANG: _005I, wing_coll_pts_coords --> _005K
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v221__005K = v433_wing_coll_pts_coords+-1*v220__005I

# op _005L expand_array_eval
# LANG: ang_vel --> _005M
# SHAPES: (1, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v222__005M = np.einsum('ad,bc->abcd', v217_ang_vel.reshape((1, 3)) ,np.ones((1, 2))).reshape((1, 1, 2, 3))

# op _006f_single_tensor_sum_with_axis_eval
# LANG: _006e --> _006g
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v240__006g = np.sum(v239__006e, axis = (3,)).reshape((1, 1, 2))

# op _008W_power_combination_eval
# LANG: _008V, _0083 --> _008X
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v329__008X = (v328__008V)*(v301__0083)
v329__008X = v329__008X.reshape((1, 12, 3))

# op _009T_power_combination_eval
# LANG: _009S, _0090 --> _009U
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v359__009U = (v358__009S)*(v331__0090)
v359__009U = v359__009U.reshape((1, 12, 3))

# op _009X_power_combination_eval
# LANG: _009W --> _009Y
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v361__009Y = (v360__009W)
v361__009Y = (v361__009Y*_009X_coeff).reshape((1, 12, 3))

# op _00aO expand_array_eval
# LANG: _00aN --> _00aP
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v388__00aP = np.einsum('ab,c->abc', v387__00aN.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _00aS cross_product_eval
# LANG: _007U, _006X --> _00aT
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v390__00aT = np.cross(v296__007U, v266__006X, axisa = 2, axisb = 2, axisc = 2)

# op _00bJ_power_combination_eval
# LANG: _00bo, _00bI --> _00bK
# SHAPES: (1, 12), (1, 12) --> (1, 12)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v417__00bK = (v406__00bo)*(v416__00bI**-1)
v417__00bK = v417__00bK.reshape((1, 12))

# op _00dW_power_combination_eval
# LANG: _00dV, _00dB --> _00dX
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v491__00dX = (v490__00dV)*(v480__00dB)
v491__00dX = v491__00dX.reshape((1, 4, 3))

# op _00eJ expand_array_eval
# LANG: _00eI --> _00eK
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v516__00eK = np.einsum('ab,c->abc', v515__00eI.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00eN cross_product_eval
# LANG: _00dr, _00cu --> _00eO
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v518__00eO = np.cross(v475__00dr, v445__00cu, axisa = 2, axisb = 2, axisc = 2)

# op _00el_power_combination_eval
# LANG: _00ek, _00e0 --> _00em
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v504__00em = (v503__00ek)*(v493__00e0)
v504__00em = v504__00em.reshape((1, 4, 3))

# op _00ep_power_combination_eval
# LANG: _00eo --> _00eq
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v506__00eq = (v505__00eo)
v506__00eq = (v506__00eq*_00ep_coeff).reshape((1, 4, 3))

# op _00f6_power_combination_eval
# LANG: _00e_, _00f5 --> _00f7
# SHAPES: (1, 4), (1, 4) --> (1, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v528__00f7 = (v524__00e_)*(v527__00f5)
v528__00f7 = v528__00f7.reshape((1, 4))

# op _005N cross_product_eval
# LANG: _005M, _005K --> _005O
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v223__005O = np.cross(v222__005M, v221__005K, axisa = 3, axisb = 3, axisc = 3)

# op _006h_power_combination_eval
# LANG: _006g --> _006i
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v241__006i = (v240__006g**0.5)
v241__006i = v241__006i.reshape((1, 1, 2))

# op _00aQ_power_combination_eval
# LANG: _00aP, _009Y --> _00aR
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v389__00aR = (v388__00aP)*(v361__009Y)
v389__00aR = v389__00aR.reshape((1, 12, 3))

# op _00aU_power_combination_eval
# LANG: _00aT --> _00aV
# SHAPES: (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v391__00aV = (v390__00aT)
v391__00aV = (v391__00aV*_00aU_coeff).reshape((1, 12, 3))

# op _00bL expand_array_eval
# LANG: _00bK --> _00bM
# SHAPES: (1, 12) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v418__00bM = np.einsum('ab,c->abc', v417__00bK.reshape((1, 12)) ,np.ones((3,))).reshape((1, 12, 3))

# op _00bP_linear_combination_eval
# LANG: _008X, _009U --> _00bQ
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v420__00bQ = v329__008X+v359__009U

# op _00eL_power_combination_eval
# LANG: _00eK, _00eq --> _00eM
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v517__00eM = (v516__00eK)*(v506__00eq)
v517__00eM = v517__00eM.reshape((1, 4, 3))

# op _00eP_power_combination_eval
# LANG: _00eO --> _00eQ
# SHAPES: (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v519__00eQ = (v518__00eO)
v519__00eQ = (v519__00eQ*_00eP_coeff).reshape((1, 4, 3))

# op _00f8 expand_array_eval
# LANG: _00f7 --> _00f9
# SHAPES: (1, 4) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v529__00f9 = np.einsum('ab,c->abc', v528__00f7.reshape((1, 4)) ,np.ones((3,))).reshape((1, 4, 3))

# op _00fc_linear_combination_eval
# LANG: _00dX, _00em --> _00fd
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v531__00fd = v491__00dX+v504__00em

# op _005P reshape_eval
# LANG: _005O --> _005Q
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v224__005Q = v223__005O.reshape((1, 2, 3))

# op _005R expand_array_eval
# LANG: frame_vel --> _005S
# SHAPES: (1, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v225__005S = np.einsum('ac,b->abc', v543_frame_vel.reshape((1, 3)) ,np.ones((2,))).reshape((1, 2, 3))

# op _006j expand_array_eval
# LANG: _006i --> _006k
# SHAPES: (1, 1, 2) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v242__006k = np.einsum('abc,d->abcd', v241__006i.reshape((1, 1, 2)) ,np.ones((3,))).reshape((1, 1, 2, 3))

# op _00bN_power_combination_eval
# LANG: _00bM, _00aV --> _00bO
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v419__00bO = (v418__00bM)*(v391__00aV)
v419__00bO = v419__00bO.reshape((1, 12, 3))

# op _00bR_linear_combination_eval
# LANG: _00bQ, _00aR --> _00bS
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v421__00bS = v420__00bQ+v389__00aR

# op _00fa_power_combination_eval
# LANG: _00f9, _00eQ --> _00fb
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v530__00fb = (v529__00f9)*(v519__00eQ)
v530__00fb = v530__00fb.reshape((1, 4, 3))

# op _00fe_linear_combination_eval
# LANG: _00fd, _00eM --> _00ff
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v532__00ff = v531__00fd+v517__00eM

# op _005U_linear_combination_eval
# LANG: _005Q, _005S --> _005V
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v227__005V = v224__005Q+v225__005S

# op _005W reshape_eval
# LANG: wing_coll_vel --> _005X
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v228__005X = v226_wing_coll_vel.reshape((1, 2, 3))

# op _006l_power_combination_eval
# LANG: _006c, _006k --> wing_bd_vtx_normals
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v536_wing_bd_vtx_normals = (v238__006c)*(v242__006k**-1)
v536_wing_bd_vtx_normals = v536_wing_bd_vtx_normals.reshape((1, 1, 2, 3))

# op _00bT_linear_combination_eval
# LANG: _00bS, _00bO --> aic_M00
# SHAPES: (1, 12, 3), (1, 12, 3) --> (1, 12, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v422_aic_M00 = v421__00bS+v419__00bO

# op _00fg_linear_combination_eval
# LANG: _00ff, _00fb --> aic_bd00
# SHAPES: (1, 4, 3), (1, 4, 3) --> (1, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v533_aic_bd00 = v532__00ff+v530__00fb

# op _005Y_linear_combination_eval
# LANG: _005V, _005X --> _005Z
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v229__005Z = v227__005V+v228__005X

# op _006A reshape_eval
# LANG: aic_M00 --> _006B
# SHAPES: (1, 12, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v253__006B = v422_aic_M00.reshape((1, 2, 6, 3))

# op _00bY reshape_eval
# LANG: wing_bd_vtx_normals --> _00bZ
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v426__00bZ = v536_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _00c7 reshape_eval
# LANG: aic_bd00 --> _00c8
# SHAPES: (1, 4, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v432__00c8 = v533_aic_bd00.reshape((1, 2, 2, 3))

# op _00fl reshape_eval
# LANG: wing_bd_vtx_normals --> _00fm
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v537__00fm = v536_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _005__linear_combination_eval
# LANG: _005Z --> wing_kinematic_vel
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v245_wing_kinematic_vel = -1*v229__005Z

# op _006C_indexed_passthrough_eval
# LANG: _006B --> aic_M
# SHAPES: (1, 2, 6, 3) --> (1, 2, 6, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v424_aic_M__temp[i_v253__006B__006C_indexed_passthrough_eval] = v253__006B.flatten()
v424_aic_M = v424_aic_M__temp.copy()

# op _006q reshape_eval
# LANG: wing_bd_vtx_normals --> _006r
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v247__006r = v536_wing_bd_vtx_normals.reshape((1, 2, 3))

# op _00b__indexed_passthrough_eval
# LANG: _00bZ --> normal_concatenated_M_mat
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v423_normal_concatenated_M_mat__temp[i_v426__00bZ__00b__indexed_passthrough_eval] = v426__00bZ.flatten()
v423_normal_concatenated_M_mat = v423_normal_concatenated_M_mat__temp.copy()

# op _00c9_indexed_passthrough_eval
# LANG: _00c8 --> aic_bd
# SHAPES: (1, 2, 2, 3) --> (1, 2, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v535_aic_bd__temp[i_v432__00c8__00c9_indexed_passthrough_eval] = v432__00c8.flatten()
v535_aic_bd = v535_aic_bd__temp.copy()

# op _00fn_indexed_passthrough_eval
# LANG: _00fm --> normal_concatenated_aic_bd_proj
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v534_normal_concatenated_aic_bd_proj__temp[i_v537__00fm__00fn_indexed_passthrough_eval] = v537__00fm.flatten()
v534_normal_concatenated_aic_bd_proj = v534_normal_concatenated_aic_bd_proj__temp.copy()

# op _006s_custom_explicit_eval
# LANG: _006r, wing_kinematic_vel --> b
# SHAPES: (1, 2, 3), (1, 2, 3) --> (1, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
temp = _006s_custom_explicit_func_b.solve(v245_wing_kinematic_vel, v247__006r)
v248_b = temp[0].copy()

# op _00c0_custom_explicit_eval
# LANG: normal_concatenated_M_mat, aic_M --> M_mat
# SHAPES: (1, 2, 3), (1, 2, 6, 3) --> (1, 2, 6)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
temp = _00c0_custom_explicit_func_M_mat.solve(v424_aic_M, v423_normal_concatenated_M_mat)
v427_M_mat = temp[0].copy()

# op _00fo_custom_explicit_eval
# LANG: normal_concatenated_aic_bd_proj, aic_bd --> aic_bd_proj
# SHAPES: (1, 2, 3), (1, 2, 2, 3) --> (1, 2, 2)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
temp = _00fo_custom_explicit_func_aic_bd_proj.solve(v535_aic_bd, v534_normal_concatenated_aic_bd_proj)
v538_aic_bd_proj = temp[0].copy()

# op _004U_indexed_passthrough_eval
# LANG: wing_gamma_w --> gamma_w
# SHAPES: (1, 3, 2) --> (1, 3, 2)
# full namespace: combine_gamma_w
v201_gamma_w__temp[i_v189_wing_gamma_w__004U_indexed_passthrough_eval] = v189_wing_gamma_w.flatten()
v201_gamma_w = v201_gamma_w__temp.copy()

# op _005f_newton_implict_eval
# LANG: gamma_w, b, aic_bd_proj, M_mat --> gamma_b
# SHAPES: (1, 3, 2), (1, 2), (1, 2, 2), (1, 2, 6) --> (1, 2)
# full namespace: solve_gamma_b_group
_005f_newton.set_guess(initial_guess_v539_gamma_b)
_005f_newton_out = _005f_newton.solve(v538_aic_bd_proj, v427_M_mat, v201_gamma_w, v248_b)
v539_gamma_b = _005f_newton_out[0]

# op _00fy_linear_combination_eval
# LANG: frame_vel --> _00fz
# SHAPES: (1, 3) --> (1, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v544__00fz = -1*v543_frame_vel

# op _00fA expand_array_eval
# LANG: _00fz --> wing_wake_kinematic_vel
# SHAPES: (1, 3) --> (1, 3, 3, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v545_wing_wake_kinematic_vel = np.einsum('ad,bc->abcd', v544__00fz.reshape((1, 3)) ,np.ones((3, 3))).reshape((1, 3, 3, 3))

# op _00fv_linear_combination_eval
# LANG: wing_wake_kinematic_vel --> wing_wake_total_vel
# SHAPES: (1, 3, 3, 3) --> (1, 3, 3, 3)
# full namespace: ComputeWakeTotalVel
v542_wing_wake_total_vel = v545_wing_wake_kinematic_vel

# op _0015_decompose_eval
# LANG: wing_wake_total_vel --> _001O, _0016
# SHAPES: (1, 3, 3, 3) --> (1, 2, 3, 3), (1, 1, 3, 3)
# full namespace: 
v52__0016 = ((v542_wing_wake_total_vel.flatten())[src_indices__0016__0015]).reshape((1, 1, 3, 3))
v75__001O = ((v542_wing_wake_total_vel.flatten())[src_indices__001O__0015]).reshape((1, 2, 3, 3))

# op _001w_power_combination_eval
# LANG: _0016 --> _001x
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v65__001x = (v52__0016)
v65__001x = (v65__001x*_001w_coeff).reshape((1, 1, 3, 3))

# op _0013_decompose_eval
# LANG: wing_bd_vtx_coords --> _0014
# SHAPES: (1, 2, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v51__0014 = ((v434_wing_bd_vtx_coords.flatten())[src_indices__0014__0013]).reshape((1, 1, 3, 3))

# op _001y_power_combination_eval
# LANG: _001x --> _001z
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v66__001z = (v65__001x)
v66__001z = (v66__001z*_001y_coeff).reshape((1, 1, 3, 3))

# op _001A_linear_combination_eval
# LANG: _0014, _001z --> _001B
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v67__001B = v51__0014+v66__001z

# op _001d_decompose_eval
# LANG: wing_wake_coords --> _001L, _001e, _001I
# SHAPES: (1, 3, 3, 3) --> (1, 2, 3, 3), (1, 1, 3, 3), (1, 2, 3, 3)
# full namespace: 
v56__001e = ((v207_wing_wake_coords.flatten())[src_indices__001e__001d]).reshape((1, 1, 3, 3))
v71__001I = ((v207_wing_wake_coords.flatten())[src_indices__001I__001d]).reshape((1, 2, 3, 3))
v73__001L = ((v207_wing_wake_coords.flatten())[src_indices__001L__001d]).reshape((1, 2, 3, 3))

# op _003l_linear_combination_eval
# LANG: _002G, _002J --> _003m
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: MeshPreprocessing_comp
v134__003m = v109__002G+-1*v111__002J

# op _001C_linear_combination_eval
# LANG: _001B, _001e --> _001D
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v68__001D = v67__001B+-1*v56__001e

# op _0024_power_combination_eval
# LANG: u --> _0025
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v89__0025 = (v79_u**2)
v89__0025 = v89__0025.reshape((1, 1))

# op _0026_power_combination_eval
# LANG: v --> _0027
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v90__0027 = (v80_v**2)
v90__0027 = v90__0027.reshape((1, 1))

# op _002C_linear_combination_eval
# LANG: wing --> _002D
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: MeshPreprocessing_comp
v107__002D = v231_wing

# op _003n pnorm_axis_eval
# LANG: _003m --> _003o
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3)
# full namespace: MeshPreprocessing_comp
v135__003o = np.sum(v134__003m**2,axis=(3,))**(1 / 2)

# op _001E reshape_eval
# LANG: _001D --> _001F
# SHAPES: (1, 1, 3, 3) --> (1, 3, 3)
# full namespace: 
v69__001F = v68__001D.reshape((1, 3, 3))

# op _0028_linear_combination_eval
# LANG: _0025, _0027 --> _0029
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v91__0029 = v89__0025+v90__0027

# op _002a_power_combination_eval
# LANG: w --> _002b
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v92__002b = (v115_w**2)
v92__002b = v92__002b.reshape((1, 1))

# op _003J_decompose_eval
# LANG: _002D --> _003P, _003K, _003L, _003O
# SHAPES: (1, 2, 3, 3) --> (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3), (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v148__003K = ((v107__002D.flatten())[src_indices__003K__003J]).reshape((1, 1, 2, 3))
v149__003L = ((v107__002D.flatten())[src_indices__003L__003J]).reshape((1, 1, 2, 3))
v151__003O = ((v107__002D.flatten())[src_indices__003O__003J]).reshape((1, 1, 2, 3))
v152__003P = ((v107__002D.flatten())[src_indices__003P__003J]).reshape((1, 1, 2, 3))

# op _003p_decompose_eval
# LANG: _003o --> _003r, _003q
# SHAPES: (1, 1, 3) --> (1, 1, 2), (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v136__003q = ((v135__003o.flatten())[src_indices__003q__003p]).reshape((1, 1, 2))
v137__003r = ((v135__003o.flatten())[src_indices__003r__003p]).reshape((1, 1, 2))

# op _0043_power_combination_eval
# LANG: _0042 --> _0044
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v161__0044 = (v160__0042)
v161__0044 = (v161__0044*_0043_coeff).reshape((1, 1, 2, 3))

# op _0046_power_combination_eval
# LANG: _0045 --> _0047
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v163__0047 = (v162__0045)
v163__0047 = (v163__0047*_0046_coeff).reshape((1, 1, 2, 3))

# op _0017_power_combination_eval
# LANG: _0016 --> _0018
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v53__0018 = (v52__0016)
v53__0018 = (v53__0018*_0017_coeff).reshape((1, 1, 3, 3))

# op _001G expand_array_eval
# LANG: _001F --> _001H
# SHAPES: (1, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v70__001H = np.einsum('acd,b->abcd', v69__001F.reshape((1, 3, 3)) ,np.ones((2,))).reshape((1, 2, 3, 3))

# op _002c_linear_combination_eval
# LANG: _0029, _002b --> v_inf_sq
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v180_v_inf_sq = v91__0029+v92__002b

# op _003M_linear_combination_eval
# LANG: _003K, _003L --> _003N
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v150__003N = v148__003K+-1*v149__003L

# op _003Q_linear_combination_eval
# LANG: _003O, _003P --> _003R
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v153__003R = v151__003O+-1*v152__003P

# op _003s_linear_combination_eval
# LANG: _003q, _003r --> _003t
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v138__003t = v136__003q+v137__003r

# op _0048_linear_combination_eval
# LANG: _0044, _0047 --> _0049
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v164__0049 = v161__0044+v163__0047

# op _004b_power_combination_eval
# LANG: _004a --> _004c
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v166__004c = (v165__004a)
v166__004c = (v166__004c*_004b_coeff).reshape((1, 1, 2, 3))

# op _00fr_decompose_eval
# LANG: gamma_b --> wing_gamma_b
# SHAPES: (1, 2) --> (1, 2)
# full namespace: seperate_gamma_b
v540_wing_gamma_b = ((v539_gamma_b.flatten())[src_indices_wing_gamma_b__00fr]).reshape((1, 2))

# op _000H_decompose_eval
# LANG: wing_gamma_b --> _000I
# SHAPES: (1, 2) --> (1, 2)
# full namespace: 
v37__000I = ((v540_wing_gamma_b.flatten())[src_indices__000I__000H]).reshape((1, 2))

# op _0019_power_combination_eval
# LANG: _0018 --> _001a
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v54__001a = (v53__0018)
v54__001a = (v54__001a*_0019_coeff).reshape((1, 1, 3, 3))

# op _001J_linear_combination_eval
# LANG: _001H, _001I --> _001K
# SHAPES: (1, 2, 3, 3), (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v72__001K = v70__001H+v71__001I

# op _003S cross_product_eval
# LANG: _003N, _003R --> _003T
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v154__003T = np.cross(v150__003N, v153__003R, axisa = 3, axisb = 3, axisc = 3)

# op _003u_power_combination_eval
# LANG: _003t --> wing_chord_length
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v139_wing_chord_length = (v138__003t)
v139_wing_chord_length = (v139_wing_chord_length*_003u_coeff).reshape((1, 1, 2))

# op _003y_linear_combination_eval
# LANG: _003w, _003x --> _003z
# SHAPES: (1, 2, 2, 3), (1, 2, 2, 3) --> (1, 2, 2, 3)
# full namespace: MeshPreprocessing_comp
v142__003z = v140__003w+-1*v141__003x

# op _004I_power_combination_eval
# LANG: v_inf_sq --> _004J
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v183__004J = (v180_v_inf_sq**0.5)
v183__004J = v183__004J.reshape((1, 1))

# op _004d_linear_combination_eval
# LANG: _0049, _004c --> _004e
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v167__004e = v164__0049+v166__004c

# op _004f_power_combination_eval
# LANG: _003e --> _004g
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v168__004g = (v130__003e)
v168__004g = (v168__004g*_004f_coeff).reshape((1, 1, 2, 3))

# op _000J reshape_eval
# LANG: _000I --> _000K
# SHAPES: (1, 2) --> (1, 1, 2)
# full namespace: 
v38__000K = v37__000I.reshape((1, 1, 2))

# op _000L_decompose_eval
# LANG: wing_gamma_w --> _000T, _000M, _000S
# SHAPES: (1, 3, 2) --> (1, 2, 2), (1, 1, 2), (1, 2, 2)
# full namespace: 
v39__000M = ((v189_wing_gamma_w.flatten())[src_indices__000M__000L]).reshape((1, 1, 2))
v42__000S = ((v189_wing_gamma_w.flatten())[src_indices__000S__000L]).reshape((1, 2, 2))
v43__000T = ((v189_wing_gamma_w.flatten())[src_indices__000T__000L]).reshape((1, 2, 2))

# op _001M_linear_combination_eval
# LANG: _001K, _001L --> _001N
# SHAPES: (1, 2, 3, 3), (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v74__001N = v72__001K+-1*v73__001L

# op _001P_power_combination_eval
# LANG: _001O --> _001Q
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v76__001Q = (v75__001O)
v76__001Q = (v76__001Q*_001P_coeff).reshape((1, 2, 3, 3))

# op _001b_linear_combination_eval
# LANG: _0014, _001a --> _001c
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v55__001c = v51__0014+v54__001a

# op _003A pnorm_axis_eval
# LANG: _003z --> _003B
# SHAPES: (1, 2, 2, 3) --> (1, 2, 2)
# full namespace: MeshPreprocessing_comp
v143__003B = np.sum(v142__003z**2,axis=(3,))**(1 / 2)

# op _003U_power_combination_eval
# LANG: _003T --> _003V
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v155__003V = (v154__003T**2)
v155__003V = v155__003V.reshape((1, 1, 2, 3))

# op _004E_single_tensor_sum_with_axis_eval
# LANG: wing_chord_length --> _004F
# SHAPES: (1, 1, 2) --> (1, 2)
# full namespace: MeshPreprocessing_comp
v181__004F = np.sum(v139_wing_chord_length, axis = (1,)).reshape((1, 2))

# op _004K_power_combination_eval
# LANG: density, _004J --> _004L
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v184__004L = (v179_density)*(v183__004J)
v184__004L = v184__004L.reshape((1, 1))

# op _004h_linear_combination_eval
# LANG: _004e, _004g --> _004i
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v169__004i = v167__004e+v168__004g

# op _004o_power_combination_eval
# LANG: _0042 --> _004p
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v172__004p = (v160__0042)
v172__004p = (v172__004p*_004o_coeff).reshape((1, 1, 2, 3))

# op _004q_power_combination_eval
# LANG: _004a --> _004r
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v173__004r = (v165__004a)
v173__004r = (v173__004r*_004q_coeff).reshape((1, 1, 2, 3))

# op _000N_linear_combination_eval
# LANG: _000K, _000M --> _000O
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: 
v40__000O = v38__000K+-1*v39__000M

# op _000U_linear_combination_eval
# LANG: _000S, _000T --> _000V
# SHAPES: (1, 2, 2), (1, 2, 2) --> (1, 2, 2)
# full namespace: 
v44__000V = v42__000S+-1*v43__000T

# op _001R_linear_combination_eval
# LANG: _001N, _001Q --> _001S
# SHAPES: (1, 2, 3, 3), (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v77__001S = v74__001N+v76__001Q

# op _001f_linear_combination_eval
# LANG: _001c, _001e --> _001g
# SHAPES: (1, 1, 3, 3), (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v57__001g = v55__001c+-1*v56__001e

# op _003C_decompose_eval
# LANG: _003B --> _003E, _003D
# SHAPES: (1, 2, 2) --> (1, 1, 2), (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v144__003D = ((v143__003B.flatten())[src_indices__003D__003C]).reshape((1, 1, 2))
v145__003E = ((v143__003B.flatten())[src_indices__003E__003C]).reshape((1, 1, 2))

# op _003W_single_tensor_sum_with_axis_eval
# LANG: _003V --> _003X
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v156__003X = np.sum(v155__003V, axis = (3,)).reshape((1, 1, 2))

# op _004G reshape_eval
# LANG: _004F --> _004H
# SHAPES: (1, 2) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v182__004H = v181__004F.reshape((1, 2, 1))

# op _004M expand_array_eval
# LANG: _004L --> _004N
# SHAPES: (1, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v185__004N = np.einsum('ac,b->abc', v184__004L.reshape((1, 1)) ,np.ones((2,))).reshape((1, 2, 1))

# op _004j reshape_eval
# LANG: _004i --> _004k
# SHAPES: (1, 1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v170__004k = v169__004i.reshape((1, 2, 3))

# op _004s_linear_combination_eval
# LANG: _004p, _004r --> _004t
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v174__004t = v172__004p+v173__004r

# op _004u_power_combination_eval
# LANG: _0045 --> _004v
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v175__004v = (v162__0045)
v175__004v = (v175__004v*_004u_coeff).reshape((1, 1, 2, 3))

# op _000P_power_combination_eval
# LANG: _000O --> _000Q
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: 
v41__000Q = (v40__000O)
v41__000Q = (v41__000Q*_000P_coeff).reshape((1, 1, 2))

# op _000W_power_combination_eval
# LANG: _000V --> _000X
# SHAPES: (1, 2, 2) --> (1, 2, 2)
# full namespace: 
v45__000X = (v44__000V)
v45__000X = (v45__000X*_000W_coeff).reshape((1, 2, 2))

# op _001T_power_combination_eval
# LANG: _001S --> _001U
# SHAPES: (1, 2, 3, 3) --> (1, 2, 3, 3)
# full namespace: 
v78__001U = (v77__001S)
v78__001U = (v78__001U*_001T_coeff).reshape((1, 2, 3, 3))

# op _001h_power_combination_eval
# LANG: _001g --> _001i
# SHAPES: (1, 1, 3, 3) --> (1, 1, 3, 3)
# full namespace: 
v58__001i = (v57__001g)
v58__001i = (v58__001i*_001h_coeff).reshape((1, 1, 3, 3))

# op _003F_linear_combination_eval
# LANG: _003D, _003E --> _003G
# SHAPES: (1, 1, 2), (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v146__003G = v144__003D+v145__003E

# op _003Y_power_combination_eval
# LANG: _003X --> _003Z
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v157__003Z = (v156__003X**0.5)
v157__003Z = v157__003Z.reshape((1, 1, 2))

# op _004O_power_combination_eval
# LANG: _004N, _004H --> _004P
# SHAPES: (1, 2, 1), (1, 2, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v186__004P = (v185__004N)*(v182__004H)
v186__004P = v186__004P.reshape((1, 2, 1))

# op _004l_linear_combination_eval
# LANG: _004k --> _004m
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v171__004m = -1*v170__004k

# op _004w_linear_combination_eval
# LANG: _004t, _004v --> _004x
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v176__004x = v174__004t+v175__004v

# op _004y_power_combination_eval
# LANG: _003e --> _004z
# SHAPES: (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v177__004z = (v130__003e)
v177__004z = (v177__004z*_004y_coeff).reshape((1, 1, 2, 3))

# op _000R_indexed_passthrough_eval
# LANG: _000Q, _000X --> wing_dgammaw_dt
# SHAPES: (1, 1, 2), (1, 2, 2) --> (1, 3, 2)
# full namespace: 
v36_wing_dgammaw_dt__temp[i_v41__000Q__000R_indexed_passthrough_eval] = v41__000Q.flatten()
v36_wing_dgammaw_dt = v36_wing_dgammaw_dt__temp.copy()
v36_wing_dgammaw_dt__temp[i_v45__000X__000R_indexed_passthrough_eval] = v45__000X.flatten()
v36_wing_dgammaw_dt = v36_wing_dgammaw_dt__temp.copy()

# op _001j_indexed_passthrough_eval
# LANG: _001i, _001U --> wing_dwake_coords_dt
# SHAPES: (1, 1, 3, 3), (1, 2, 3, 3) --> (1, 3, 3, 3)
# full namespace: 
v50_wing_dwake_coords_dt__temp[i_v58__001i__001j_indexed_passthrough_eval] = v58__001i.flatten()
v50_wing_dwake_coords_dt = v50_wing_dwake_coords_dt__temp.copy()
v50_wing_dwake_coords_dt__temp[i_v78__001U__001j_indexed_passthrough_eval] = v78__001U.flatten()
v50_wing_dwake_coords_dt = v50_wing_dwake_coords_dt__temp.copy()

# op _002q_linear_combination_eval
# LANG: theta, gamma --> alpha
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v100_alpha = v85_theta+-1*v87_gamma

# op _002s_linear_combination_eval
# LANG: psi, psiw --> beta
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v101_beta = v86_psi+v88_psiw

# op _003H_power_combination_eval
# LANG: _003G --> wing_span_length
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v147_wing_span_length = (v146__003G)
v147_wing_span_length = (v147_wing_span_length*_003H_coeff).reshape((1, 1, 2))

# op _003__power_combination_eval
# LANG: _003Z --> wing_s_panel
# SHAPES: (1, 1, 2) --> (1, 1, 2)
# full namespace: MeshPreprocessing_comp
v158_wing_s_panel = (v157__003Z)
v158_wing_s_panel = (v158_wing_s_panel*_003__coeff).reshape((1, 1, 2))

# op _004A_linear_combination_eval
# LANG: _004x, _004z --> wing_eval_pts_coords
# SHAPES: (1, 1, 2, 3), (1, 1, 2, 3) --> (1, 1, 2, 3)
# full namespace: MeshPreprocessing_comp
v178_wing_eval_pts_coords = v176__004x+v177__004z

# op _004Q_power_combination_eval
# LANG: _004P --> wing_re_span
# SHAPES: (1, 2, 1) --> (1, 2, 1)
# full namespace: MeshPreprocessing_comp
v187_wing_re_span = (v186__004P)
v187_wing_re_span = (v187_wing_re_span*_004Q_coeff).reshape((1, 2, 1))

# op _004n_indexed_passthrough_eval
# LANG: _004m --> bd_vec
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: MeshPreprocessing_comp
v159_bd_vec__temp[i_v171__004m__004n_indexed_passthrough_eval] = v171__004m.flatten()
v159_bd_vec = v159_bd_vec__temp.copy()

# op _006u_indexed_passthrough_eval
# LANG: _006r --> normal_concatenated_b
# SHAPES: (1, 2, 3) --> (1, 2, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v244_normal_concatenated_b__temp[i_v247__006r__006u_indexed_passthrough_eval] = v247__006r.flatten()
v244_normal_concatenated_b = v244_normal_concatenated_b__temp.copy()