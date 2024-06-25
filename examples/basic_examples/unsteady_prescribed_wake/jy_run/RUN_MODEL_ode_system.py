

# RUN_MODEL_ode_system

# system evaluation block

# op _005g_linear_combination_eval
# LANG: u --> _005h
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v198__005h = -1*v174_u

# op _005j_linear_combination_eval
# LANG: w --> _005k
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v199__005k = -1*v210_w

# op _005i_indexed_passthrough_eval
# LANG: _005h, _005k --> frame_vel
# SHAPES: (1, 1), (1, 1) --> (1, 3)
# full namespace: adapter_comp
v639_frame_vel__temp[i_v198__005h__005i_indexed_passthrough_eval] = v198__005h.flatten()
v639_frame_vel = v639_frame_vel__temp.copy()
v639_frame_vel__temp[i_v199__005k__005i_indexed_passthrough_eval] = v199__005k.flatten()
v639_frame_vel = v639_frame_vel__temp.copy()

# op _005D_decompose_eval
# LANG: frame_vel --> _005I, _005E
# SHAPES: (1, 3) --> (1, 1), (1, 1)
# full namespace: MeshPreprocessing_comp
v212__005E = ((v639_frame_vel.flatten())[src_indices__005E__005D]).reshape((1, 1))
v214__005I = ((v639_frame_vel.flatten())[src_indices__005I__005D]).reshape((1, 1))

# op _005F_linear_combination_eval
# LANG: _005E --> _005G
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v213__005G = -1*v212__005E

# op _005J_linear_combination_eval
# LANG: _005I --> _005K
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v215__005K = -1*v214__005I

# op _005H_indexed_passthrough_eval
# LANG: _005G, _005K, w --> fs
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v211_fs__temp[i_v213__005G__005H_indexed_passthrough_eval] = v213__005G.flatten()
v211_fs = v211_fs__temp.copy()
v211_fs__temp[i_v215__005K__005H_indexed_passthrough_eval] = v215__005K.flatten()
v211_fs = v211_fs__temp.copy()
v211_fs__temp[i_v210_w__005H_indexed_passthrough_eval] = v210_w.flatten()
v211_fs = v211_fs__temp.copy()

# op _005q_decompose_eval
# LANG: eel --> _0060, _005r, _005u, _005R, _005U, _005V, _005_, _006i, _006j, _006P, _006S, _006X
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 5, 3), (1, 40, 5, 3), (1, 1, 5, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 41, 4, 3), (1, 41, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v204__005r = ((v327_eel.flatten())[src_indices__005r__005q]).reshape((1, 40, 5, 3))
v206__005u = ((v327_eel.flatten())[src_indices__005u__005q]).reshape((1, 40, 5, 3))
v219__005R = ((v327_eel.flatten())[src_indices__005R__005q]).reshape((1, 1, 5, 3))
v221__005U = ((v327_eel.flatten())[src_indices__005U__005q]).reshape((1, 40, 4, 3))
v222__005V = ((v327_eel.flatten())[src_indices__005V__005q]).reshape((1, 40, 4, 3))
v225__005_ = ((v327_eel.flatten())[src_indices__005___005q]).reshape((1, 40, 4, 3))
v226__0060 = ((v327_eel.flatten())[src_indices__0060__005q]).reshape((1, 40, 4, 3))
v236__006i = ((v327_eel.flatten())[src_indices__006i__005q]).reshape((1, 41, 4, 3))
v237__006j = ((v327_eel.flatten())[src_indices__006j__005q]).reshape((1, 41, 4, 3))
v256__006P = ((v327_eel.flatten())[src_indices__006P__005q]).reshape((1, 40, 4, 3))
v258__006S = ((v327_eel.flatten())[src_indices__006S__005q]).reshape((1, 40, 4, 3))
v261__006X = ((v327_eel.flatten())[src_indices__006X__005q]).reshape((1, 40, 4, 3))

# op _008b_decompose_eval
# LANG: eel_wake_coords --> _008c
# SHAPES: (1, 69, 5, 3) --> (1, 1, 5, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v305__008c = ((v303_eel_wake_coords.flatten())[src_indices__008c__008b]).reshape((1, 1, 5, 3))

# op _005L_power_combination_eval
# LANG: fs --> _005M
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v216__005M = (v211_fs)
v216__005M = (v216__005M*_005L_coeff).reshape((1, 3))

# op _005W_linear_combination_eval
# LANG: _005U, _005V --> _005X
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v223__005X = v221__005U+v222__005V

# op _0061_linear_combination_eval
# LANG: _0060, _005_ --> _0062
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v227__0062 = v225__005_+v226__0060

# op _008d_power_combination_eval
# LANG: _008c --> _008e
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v306__008e = (v305__008c)
v306__008e = v306__008e.reshape((1, 1, 5, 3))

# op _005N_power_combination_eval
# LANG: _005M --> _005O
# SHAPES: (1, 3) --> (1, 3)
# full namespace: MeshPreprocessing_comp
v217__005O = (v216__005M)
v217__005O = (v217__005O*_005N_coeff).reshape((1, 3))

# op _005Y_power_combination_eval
# LANG: _005X --> _005Z
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v224__005Z = (v223__005X)
v224__005Z = (v224__005Z*_005Y_coeff).reshape((1, 40, 4, 3))

# op _0063_power_combination_eval
# LANG: _0062 --> _0064
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v228__0064 = (v227__0062)
v228__0064 = (v228__0064*_0063_coeff).reshape((1, 40, 4, 3))

# op _008a_indexed_passthrough_eval
# LANG: _008e, eel_wake_coords --> eel_TE_wake_coords
# SHAPES: (1, 1, 5, 3), (1, 69, 5, 3) --> (1, 70, 5, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group
v351_eel_TE_wake_coords__temp[i_v303_eel_wake_coords__008a_indexed_passthrough_eval] = v303_eel_wake_coords.flatten()
v351_eel_TE_wake_coords = v351_eel_TE_wake_coords__temp.copy()
v351_eel_TE_wake_coords__temp[i_v306__008e__008a_indexed_passthrough_eval] = v306__008e.flatten()
v351_eel_TE_wake_coords = v351_eel_TE_wake_coords__temp.copy()

# op _005P expand_array_eval
# LANG: _005O --> _005Q
# SHAPES: (1, 3) --> (1, 1, 5, 3)
# full namespace: MeshPreprocessing_comp
v218__005Q = np.einsum('ad,bc->abcd', v217__005O.reshape((1, 3)) ,np.ones((1, 5))).reshape((1, 1, 5, 3))

# op _005s_power_combination_eval
# LANG: _005r --> _005t
# SHAPES: (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v205__005t = (v204__005r)
v205__005t = (v205__005t*_005s_coeff).reshape((1, 40, 5, 3))

# op _005v_power_combination_eval
# LANG: _005u --> _005w
# SHAPES: (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v207__005w = (v206__005u)
v207__005w = (v207__005w*_005v_coeff).reshape((1, 40, 5, 3))

# op _0065_linear_combination_eval
# LANG: _005Z, _0064 --> eel_coll_pts_coords
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v529_eel_coll_pts_coords = v224__005Z+v228__0064

# op _009r_decompose_eval
# LANG: eel_TE_wake_coords --> _009s, _009t, _009u, _009v
# SHAPES: (1, 70, 5, 3) --> (1, 69, 4, 3), (1, 69, 4, 3), (1, 69, 4, 3), (1, 69, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v352__009s = ((v351_eel_TE_wake_coords.flatten())[src_indices__009s__009r]).reshape((1, 69, 4, 3))
v353__009t = ((v351_eel_TE_wake_coords.flatten())[src_indices__009t__009r]).reshape((1, 69, 4, 3))
v354__009u = ((v351_eel_TE_wake_coords.flatten())[src_indices__009u__009r]).reshape((1, 69, 4, 3))
v355__009v = ((v351_eel_TE_wake_coords.flatten())[src_indices__009v__009r]).reshape((1, 69, 4, 3))

# op _005S_linear_combination_eval
# LANG: _005R, _005Q --> _005T
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: MeshPreprocessing_comp
v220__005T = v219__005R+v218__005Q

# op _005x_linear_combination_eval
# LANG: _005t, _005w --> _005y
# SHAPES: (1, 40, 5, 3), (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v208__005y = v205__005t+v207__005w

# op _009C reshape_eval
# LANG: _009s --> _009D
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v359__009D = v352__009s.reshape((1, 276, 3))

# op _009Q reshape_eval
# LANG: eel_coll_pts_coords --> _009R
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v366__009R = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _009W reshape_eval
# LANG: _009t --> _009X
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v369__009X = v353__009t.reshape((1, 276, 3))

# op _009w reshape_eval
# LANG: eel_coll_pts_coords --> _009x
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v356__009x = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00a9 reshape_eval
# LANG: eel_coll_pts_coords --> _00aa
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v376__00aa = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00af reshape_eval
# LANG: _009u --> _00ag
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v379__00ag = v354__009u.reshape((1, 276, 3))

# op _005z_indexed_passthrough_eval
# LANG: _005y, _005T --> eel_bd_vtx_coords
# SHAPES: (1, 40, 5, 3), (1, 1, 5, 3) --> (1, 41, 5, 3)
# full namespace: MeshPreprocessing_comp
v530_eel_bd_vtx_coords__temp[i_v208__005y__005z_indexed_passthrough_eval] = v208__005y.flatten()
v530_eel_bd_vtx_coords = v530_eel_bd_vtx_coords__temp.copy()
v530_eel_bd_vtx_coords__temp[i_v220__005T__005z_indexed_passthrough_eval] = v220__005T.flatten()
v530_eel_bd_vtx_coords = v530_eel_bd_vtx_coords__temp.copy()

# op _009E expand_array_eval
# LANG: _009D --> _009F
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v360__009F = np.einsum('acd,b->abcd', v359__009D.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _009S expand_array_eval
# LANG: _009R --> _009T
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v367__009T = np.einsum('abd,c->abcd', v366__009R.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _009Y expand_array_eval
# LANG: _009X --> _009Z
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v370__009Z = np.einsum('acd,b->abcd', v369__009X.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _009y expand_array_eval
# LANG: _009x --> _009z
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v357__009z = np.einsum('abd,c->abcd', v356__009x.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00ab expand_array_eval
# LANG: _00aa --> _00ac
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v377__00ac = np.einsum('abd,c->abcd', v376__00aa.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00ah expand_array_eval
# LANG: _00ag --> _00ai
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v380__00ai = np.einsum('acd,b->abcd', v379__00ag.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _00at reshape_eval
# LANG: eel_coll_pts_coords --> _00au
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v386__00au = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00az reshape_eval
# LANG: _009v --> _00aA
# SHAPES: (1, 69, 4, 3) --> (1, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v389__00aA = v355__009v.reshape((1, 276, 3))

# op _009A reshape_eval
# LANG: _009z --> _009B
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v358__009B = v357__009z.reshape((1, 44160, 3))

# op _009G reshape_eval
# LANG: _009F --> _009H
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v361__009H = v360__009F.reshape((1, 44160, 3))

# op _009U reshape_eval
# LANG: _009T --> _009V
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v368__009V = v367__009T.reshape((1, 44160, 3))

# op _009_ reshape_eval
# LANG: _009Z --> _00a0
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v371__00a0 = v370__009Z.reshape((1, 44160, 3))

# op _00aB expand_array_eval
# LANG: _00aA --> _00aC
# SHAPES: (1, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v390__00aC = np.einsum('acd,b->abcd', v389__00aA.reshape((1, 276, 3)) ,np.ones((160,))).reshape((1, 160, 276, 3))

# op _00ad reshape_eval
# LANG: _00ac --> _00ae
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v378__00ae = v377__00ac.reshape((1, 44160, 3))

# op _00aj reshape_eval
# LANG: _00ai --> _00ak
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v381__00ak = v380__00ai.reshape((1, 44160, 3))

# op _00av expand_array_eval
# LANG: _00au --> _00aw
# SHAPES: (1, 160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v387__00aw = np.einsum('abd,c->abcd', v386__00au.reshape((1, 160, 3)) ,np.ones((276,))).reshape((1, 160, 276, 3))

# op _00eZ_decompose_eval
# LANG: eel_bd_vtx_coords --> _00e_, _00f0, _00f1, _00f2
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v531__00e_ = ((v530_eel_bd_vtx_coords.flatten())[src_indices__00e___00eZ]).reshape((1, 40, 4, 3))
v532__00f0 = ((v530_eel_bd_vtx_coords.flatten())[src_indices__00f0__00eZ]).reshape((1, 40, 4, 3))
v533__00f1 = ((v530_eel_bd_vtx_coords.flatten())[src_indices__00f1__00eZ]).reshape((1, 40, 4, 3))
v534__00f2 = ((v530_eel_bd_vtx_coords.flatten())[src_indices__00f2__00eZ]).reshape((1, 40, 4, 3))

# op _009I_linear_combination_eval
# LANG: _009B, _009H --> _009J
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v362__009J = v358__009B+-1*v361__009H

# op _00a1_linear_combination_eval
# LANG: _009V, _00a0 --> _00a2
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v372__00a2 = v368__009V+-1*v371__00a0

# op _00aD reshape_eval
# LANG: _00aC --> _00aE
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v391__00aE = v390__00aC.reshape((1, 44160, 3))

# op _00al_linear_combination_eval
# LANG: _00ae, _00ak --> _00am
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v382__00am = v378__00ae+-1*v381__00ak

# op _00ax reshape_eval
# LANG: _00aw --> _00ay
# SHAPES: (1, 160, 276, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v388__00ay = v387__00aw.reshape((1, 44160, 3))

# op _00f3 reshape_eval
# LANG: eel_coll_pts_coords --> _00f4
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v535__00f4 = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00f9 reshape_eval
# LANG: _00e_ --> _00fa
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v538__00fa = v531__00e_.reshape((1, 160, 3))

# op _00fH reshape_eval
# LANG: eel_coll_pts_coords --> _00fI
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v555__00fI = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00fN reshape_eval
# LANG: _00f1 --> _00fO
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v558__00fO = v533__00f1.reshape((1, 160, 3))

# op _00fn reshape_eval
# LANG: eel_coll_pts_coords --> _00fo
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v545__00fo = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00ft reshape_eval
# LANG: _00f0 --> _00fu
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v548__00fu = v532__00f0.reshape((1, 160, 3))

# op _009K_power_combination_eval
# LANG: _009J --> _009L
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v363__009L = (v362__009J**2)
v363__009L = v363__009L.reshape((1, 44160, 3))

# op _00a3_power_combination_eval
# LANG: _00a2 --> _00a4
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v373__00a4 = (v372__00a2**2)
v373__00a4 = v373__00a4.reshape((1, 44160, 3))

# op _00aF_linear_combination_eval
# LANG: _00ay, _00aE --> _00aG
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v392__00aG = v388__00ay+-1*v391__00aE

# op _00an_power_combination_eval
# LANG: _00am --> _00ao
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v383__00ao = (v382__00am**2)
v383__00ao = v383__00ao.reshape((1, 44160, 3))

# op _00f5 expand_array_eval
# LANG: _00f4 --> _00f6
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v536__00f6 = np.einsum('abd,c->abcd', v535__00f4.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00fJ expand_array_eval
# LANG: _00fI --> _00fK
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v556__00fK = np.einsum('abd,c->abcd', v555__00fI.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00fP expand_array_eval
# LANG: _00fO --> _00fQ
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v559__00fQ = np.einsum('acd,b->abcd', v558__00fO.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00fb expand_array_eval
# LANG: _00fa --> _00fc
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v539__00fc = np.einsum('acd,b->abcd', v538__00fa.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00fp expand_array_eval
# LANG: _00fo --> _00fq
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v546__00fq = np.einsum('abd,c->abcd', v545__00fo.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00fv expand_array_eval
# LANG: _00fu --> _00fw
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v549__00fw = np.einsum('acd,b->abcd', v548__00fu.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00g0 reshape_eval
# LANG: eel_coll_pts_coords --> _00g1
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v565__00g1 = v529_eel_coll_pts_coords.reshape((1, 160, 3))

# op _00g6 reshape_eval
# LANG: _00f2 --> _00g7
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v568__00g7 = v534__00f2.reshape((1, 160, 3))

# op _009M_single_tensor_sum_with_axis_eval
# LANG: _009L --> _009N
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v364__009N = np.sum(v363__009L, axis = (2,)).reshape((1, 44160))

# op _00a5_single_tensor_sum_with_axis_eval
# LANG: _00a4 --> _00a6
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v374__00a6 = np.sum(v373__00a4, axis = (2,)).reshape((1, 44160))

# op _00aH_power_combination_eval
# LANG: _00aG --> _00aI
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v393__00aI = (v392__00aG**2)
v393__00aI = v393__00aI.reshape((1, 44160, 3))

# op _00ap_single_tensor_sum_with_axis_eval
# LANG: _00ao --> _00aq
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v384__00aq = np.sum(v383__00ao, axis = (2,)).reshape((1, 44160))

# op _00f7 reshape_eval
# LANG: _00f6 --> _00f8
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v537__00f8 = v536__00f6.reshape((1, 25600, 3))

# op _00fL reshape_eval
# LANG: _00fK --> _00fM
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v557__00fM = v556__00fK.reshape((1, 25600, 3))

# op _00fR reshape_eval
# LANG: _00fQ --> _00fS
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v560__00fS = v559__00fQ.reshape((1, 25600, 3))

# op _00fd reshape_eval
# LANG: _00fc --> _00fe
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v540__00fe = v539__00fc.reshape((1, 25600, 3))

# op _00fr reshape_eval
# LANG: _00fq --> _00fs
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v547__00fs = v546__00fq.reshape((1, 25600, 3))

# op _00fx reshape_eval
# LANG: _00fw --> _00fy
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v550__00fy = v549__00fw.reshape((1, 25600, 3))

# op _00g2 expand_array_eval
# LANG: _00g1 --> _00g3
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v566__00g3 = np.einsum('abd,c->abcd', v565__00g1.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _00g8 expand_array_eval
# LANG: _00g7 --> _00g9
# SHAPES: (1, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v569__00g9 = np.einsum('acd,b->abcd', v568__00g7.reshape((1, 160, 3)) ,np.ones((160,))).reshape((1, 160, 160, 3))

# op _009O_power_combination_eval
# LANG: _009N --> _009P
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v365__009P = (v364__009N**0.5)
v365__009P = v365__009P.reshape((1, 44160))

# op _00a7_power_combination_eval
# LANG: _00a6 --> _00a8
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v375__00a8 = (v374__00a6**0.5)
v375__00a8 = v375__00a8.reshape((1, 44160))

# op _00aJ_single_tensor_sum_with_axis_eval
# LANG: _00aI --> _00aK
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v394__00aK = np.sum(v393__00aI, axis = (2,)).reshape((1, 44160))

# op _00ar_power_combination_eval
# LANG: _00aq --> _00as
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v385__00as = (v384__00aq**0.5)
v385__00as = v385__00as.reshape((1, 44160))

# op _00fT_linear_combination_eval
# LANG: _00fM, _00fS --> _00fU
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v561__00fU = v557__00fM+-1*v560__00fS

# op _00ff_linear_combination_eval
# LANG: _00f8, _00fe --> _00fg
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v541__00fg = v537__00f8+-1*v540__00fe

# op _00fz_linear_combination_eval
# LANG: _00fs, _00fy --> _00fA
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v551__00fA = v547__00fs+-1*v550__00fy

# op _00g4 reshape_eval
# LANG: _00g3 --> _00g5
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v567__00g5 = v566__00g3.reshape((1, 25600, 3))

# op _00ga reshape_eval
# LANG: _00g9 --> _00gb
# SHAPES: (1, 160, 160, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v570__00gb = v569__00g9.reshape((1, 25600, 3))

# op _00aL_power_combination_eval
# LANG: _00aK --> _00aM
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v395__00aM = (v394__00aK**0.5)
v395__00aM = v395__00aM.reshape((1, 44160))

# op _00aR_power_combination_eval
# LANG: _009J, _00a2 --> _00aS
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v398__00aS = (v362__009J)*(v372__00a2)
v398__00aS = v398__00aS.reshape((1, 44160, 3))

# op _00aV_power_combination_eval
# LANG: _009P --> _00aW
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v400__00aW = (v365__009P**2)
v400__00aW = v400__00aW.reshape((1, 44160))

# op _00aX_power_combination_eval
# LANG: _00a8 --> _00aY
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v401__00aY = (v375__00a8**2)
v401__00aY = v401__00aY.reshape((1, 44160))

# op _00bO_power_combination_eval
# LANG: _00am, _00a2 --> _00bP
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v428__00bP = (v372__00a2)*(v382__00am)
v428__00bP = v428__00bP.reshape((1, 44160, 3))

# op _00bS_power_combination_eval
# LANG: _00a8 --> _00bT
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v430__00bT = (v375__00a8**2)
v430__00bT = v430__00bT.reshape((1, 44160))

# op _00bU_power_combination_eval
# LANG: _00as --> _00bV
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v431__00bV = (v385__00as**2)
v431__00bV = v431__00bV.reshape((1, 44160))

# op _00bs_power_combination_eval
# LANG: _009P --> _00bt
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v417__00bt = (v365__009P)
v417__00bt = (v417__00bt*_00bs_coeff).reshape((1, 44160))

# op _00cp_power_combination_eval
# LANG: _00a8 --> _00cq
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v447__00cq = (v375__00a8)
v447__00cq = (v447__00cq*_00cp_coeff).reshape((1, 44160))

# op _00fB_power_combination_eval
# LANG: _00fA --> _00fC
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v552__00fC = (v551__00fA**2)
v552__00fC = v552__00fC.reshape((1, 25600, 3))

# op _00fV_power_combination_eval
# LANG: _00fU --> _00fW
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v562__00fW = (v561__00fU**2)
v562__00fW = v562__00fW.reshape((1, 25600, 3))

# op _00fh_power_combination_eval
# LANG: _00fg --> _00fi
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v542__00fi = (v541__00fg**2)
v542__00fi = v542__00fi.reshape((1, 25600, 3))

# op _00gc_linear_combination_eval
# LANG: _00g5, _00gb --> _00gd
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v571__00gd = v567__00g5+-1*v570__00gb

# op _00aT_single_tensor_sum_with_axis_eval
# LANG: _00aS --> _00aU
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v399__00aU = np.sum(v398__00aS, axis = (2,)).reshape((1, 44160))

# op _00b0_linear_combination_eval
# LANG: _00aW --> _00b1
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v403__00b1 = _00b0_constant+v400__00aW

# op _00bQ_single_tensor_sum_with_axis_eval
# LANG: _00bP --> _00bR
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v429__00bR = np.sum(v428__00bP, axis = (2,)).reshape((1, 44160))

# op _00bY_linear_combination_eval
# LANG: _00bT --> _00bZ
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v433__00bZ = _00bY_constant+v430__00bT

# op _00ba_linear_combination_eval
# LANG: _00aY --> _00bb
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v408__00bb = _00ba_constant+v401__00aY

# op _00bq_linear_combination_eval
# LANG: _00aW, _00aY --> _00br
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v416__00br = v400__00aW+v401__00aY

# op _00bu_power_combination_eval
# LANG: _00a8, _00bt --> _00bv
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v418__00bv = (v417__00bt)*(v375__00a8)
v418__00bv = v418__00bv.reshape((1, 44160))

# op _00c7_linear_combination_eval
# LANG: _00bV --> _00c8
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v438__00c8 = _00c7_constant+v431__00bV

# op _00cL_power_combination_eval
# LANG: _00aG, _00am --> _00cM
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v458__00cM = (v382__00am)*(v392__00aG)
v458__00cM = v458__00cM.reshape((1, 44160, 3))

# op _00cP_power_combination_eval
# LANG: _00as --> _00cQ
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v460__00cQ = (v385__00as**2)
v460__00cQ = v460__00cQ.reshape((1, 44160))

# op _00cR_power_combination_eval
# LANG: _00aM --> _00cS
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v461__00cS = (v395__00aM**2)
v461__00cS = v461__00cS.reshape((1, 44160))

# op _00cn_linear_combination_eval
# LANG: _00bT, _00bV --> _00co
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v446__00co = v430__00bT+v431__00bV

# op _00cr_power_combination_eval
# LANG: _00as, _00cq --> _00cs
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v448__00cs = (v447__00cq)*(v385__00as)
v448__00cs = v448__00cs.reshape((1, 44160))

# op _00dm_power_combination_eval
# LANG: _00as --> _00dn
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v477__00dn = (v385__00as)
v477__00dn = (v477__00dn*_00dm_coeff).reshape((1, 44160))

# op _00fD_single_tensor_sum_with_axis_eval
# LANG: _00fC --> _00fE
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v553__00fE = np.sum(v552__00fC, axis = (2,)).reshape((1, 25600))

# op _00fX_single_tensor_sum_with_axis_eval
# LANG: _00fW --> _00fY
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v563__00fY = np.sum(v562__00fW, axis = (2,)).reshape((1, 25600))

# op _00fj_single_tensor_sum_with_axis_eval
# LANG: _00fi --> _00fk
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v543__00fk = np.sum(v542__00fi, axis = (2,)).reshape((1, 25600))

# op _00ge_power_combination_eval
# LANG: _00gd --> _00gf
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v572__00gf = (v571__00gd**2)
v572__00gf = v572__00gf.reshape((1, 25600, 3))

# op _00b2_linear_combination_eval
# LANG: _00b1 --> _00b3
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v404__00b3 = _00b2_constant+v403__00b1

# op _00b__linear_combination_eval
# LANG: _00bZ --> _00c0
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v434__00c0 = _00b__constant+v433__00bZ

# op _00bc_linear_combination_eval
# LANG: _00bb --> _00bd
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v409__00bd = _00bc_constant+v408__00bb

# op _00bk_power_combination_eval
# LANG: _00aW, _00aY --> _00bl
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v413__00bl = (v400__00aW)*(v401__00aY)
v413__00bl = v413__00bl.reshape((1, 44160))

# op _00bm_power_combination_eval
# LANG: _00aU --> _00bn
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v414__00bn = (v399__00aU**2)
v414__00bn = v414__00bn.reshape((1, 44160))

# op _00bw_linear_combination_eval
# LANG: _00br, _00bv --> _00bx
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v419__00bx = v416__00br+-1*v418__00bv

# op _00c9_linear_combination_eval
# LANG: _00c8 --> _00ca
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v439__00ca = _00c9_constant+v438__00c8

# op _00cN_single_tensor_sum_with_axis_eval
# LANG: _00cM --> _00cO
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v459__00cO = np.sum(v458__00cM, axis = (2,)).reshape((1, 44160))

# op _00cV_linear_combination_eval
# LANG: _00cQ --> _00cW
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v463__00cW = _00cV_constant+v460__00cQ

# op _00ch_power_combination_eval
# LANG: _00bT, _00bV --> _00ci
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v443__00ci = (v430__00bT)*(v431__00bV)
v443__00ci = v443__00ci.reshape((1, 44160))

# op _00cj_power_combination_eval
# LANG: _00bR --> _00ck
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v444__00ck = (v429__00bR**2)
v444__00ck = v444__00ck.reshape((1, 44160))

# op _00ct_linear_combination_eval
# LANG: _00co, _00cs --> _00cu
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v449__00cu = v446__00co+-1*v448__00cs

# op _00d4_linear_combination_eval
# LANG: _00cS --> _00d5
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v468__00d5 = _00d4_constant+v461__00cS

# op _00dI_power_combination_eval
# LANG: _00aG, _009J --> _00dJ
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v488__00dJ = (v392__00aG)*(v362__009J)
v488__00dJ = v488__00dJ.reshape((1, 44160, 3))

# op _00dM_power_combination_eval
# LANG: _00aM --> _00dN
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v490__00dN = (v395__00aM**2)
v490__00dN = v490__00dN.reshape((1, 44160))

# op _00dO_power_combination_eval
# LANG: _009P --> _00dP
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v491__00dP = (v365__009P**2)
v491__00dP = v491__00dP.reshape((1, 44160))

# op _00dk_linear_combination_eval
# LANG: _00cQ, _00cS --> _00dl
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v476__00dl = v460__00cQ+v461__00cS

# op _00do_power_combination_eval
# LANG: _00aM, _00dn --> _00dp
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v478__00dp = (v477__00dn)*(v395__00aM)
v478__00dp = v478__00dp.reshape((1, 44160))

# op _00ej_power_combination_eval
# LANG: _00aM --> _00ek
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v507__00ek = (v395__00aM)
v507__00ek = (v507__00ek*_00ej_coeff).reshape((1, 44160))

# op _00fF_power_combination_eval
# LANG: _00fE --> _00fG
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v554__00fG = (v553__00fE**0.5)
v554__00fG = v554__00fG.reshape((1, 25600))

# op _00fZ_power_combination_eval
# LANG: _00fY --> _00f_
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v564__00f_ = (v563__00fY**0.5)
v564__00f_ = v564__00f_.reshape((1, 25600))

# op _00fl_power_combination_eval
# LANG: _00fk --> _00fm
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v544__00fm = (v543__00fk**0.5)
v544__00fm = v544__00fm.reshape((1, 25600))

# op _00gO_power_combination_eval
# LANG: _00fU, _00fA --> _00gP
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v590__00gP = (v551__00fA)*(v561__00fU)
v590__00gP = v590__00gP.reshape((1, 25600, 3))

# op _00gg_single_tensor_sum_with_axis_eval
# LANG: _00gf --> _00gh
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v573__00gh = np.sum(v572__00gf, axis = (2,)).reshape((1, 25600))

# op _00go_power_combination_eval
# LANG: _00fg, _00fA --> _00gp
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v577__00gp = (v541__00fg)*(v551__00fA)
v577__00gp = v577__00gp.reshape((1, 25600, 3))

# op _00aZ_linear_combination_eval
# LANG: _00aW, _00aU --> _00a_
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v402__00a_ = v400__00aW+-1*v399__00aU

# op _00b4_power_combination_eval
# LANG: _00b3 --> _00b5
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v405__00b5 = (v404__00b3**0.5)
v405__00b5 = v405__00b5.reshape((1, 44160))

# op _00b8_linear_combination_eval
# LANG: _00aY, _00aU --> _00b9
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v407__00b9 = v401__00aY+-1*v399__00aU

# op _00bW_linear_combination_eval
# LANG: _00bT, _00bR --> _00bX
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v432__00bX = v430__00bT+-1*v429__00bR

# op _00be_power_combination_eval
# LANG: _00bd --> _00bf
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v410__00bf = (v409__00bd**0.5)
v410__00bf = v410__00bf.reshape((1, 44160))

# op _00bo_linear_combination_eval
# LANG: _00bl, _00bn --> _00bp
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v415__00bp = v413__00bl+-1*v414__00bn

# op _00by_power_combination_eval
# LANG: _00bx --> _00bz
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v420__00bz = (v419__00bx)
v420__00bz = (v420__00bz*_00by_coeff).reshape((1, 44160))

# op _00c1_power_combination_eval
# LANG: _00c0 --> _00c2
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v435__00c2 = (v434__00c0**0.5)
v435__00c2 = v435__00c2.reshape((1, 44160))

# op _00c5_linear_combination_eval
# LANG: _00bV, _00bR --> _00c6
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v437__00c6 = v431__00bV+-1*v429__00bR

# op _00cX_linear_combination_eval
# LANG: _00cW --> _00cY
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v464__00cY = _00cX_constant+v463__00cW

# op _00cb_power_combination_eval
# LANG: _00ca --> _00cc
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v440__00cc = (v439__00ca**0.5)
v440__00cc = v440__00cc.reshape((1, 44160))

# op _00cl_linear_combination_eval
# LANG: _00ci, _00ck --> _00cm
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v445__00cm = v443__00ci+-1*v444__00ck

# op _00cv_power_combination_eval
# LANG: _00cu --> _00cw
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v450__00cw = (v449__00cu)
v450__00cw = (v450__00cw*_00cv_coeff).reshape((1, 44160))

# op _00d6_linear_combination_eval
# LANG: _00d5 --> _00d7
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v469__00d7 = _00d6_constant+v468__00d5

# op _00dK_single_tensor_sum_with_axis_eval
# LANG: _00dJ --> _00dL
# SHAPES: (1, 44160, 3) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v489__00dL = np.sum(v488__00dJ, axis = (2,)).reshape((1, 44160))

# op _00dS_linear_combination_eval
# LANG: _00dN --> _00dT
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v493__00dT = _00dS_constant+v490__00dN

# op _00de_power_combination_eval
# LANG: _00cQ, _00cS --> _00df
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v473__00df = (v460__00cQ)*(v461__00cS)
v473__00df = v473__00df.reshape((1, 44160))

# op _00dg_power_combination_eval
# LANG: _00cO --> _00dh
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v474__00dh = (v459__00cO**2)
v474__00dh = v474__00dh.reshape((1, 44160))

# op _00dq_linear_combination_eval
# LANG: _00dl, _00dp --> _00dr
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v479__00dr = v476__00dl+-1*v478__00dp

# op _00e1_linear_combination_eval
# LANG: _00dP --> _00e2
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v498__00e2 = _00e1_constant+v491__00dP

# op _00eh_linear_combination_eval
# LANG: _00dN, _00dP --> _00ei
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v506__00ei = v490__00dN+v491__00dP

# op _00el_power_combination_eval
# LANG: _00ek, _009P --> _00em
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v508__00em = (v507__00ek)*(v365__009P)
v508__00em = v508__00em.reshape((1, 44160))

# op _00gQ_single_tensor_sum_with_axis_eval
# LANG: _00gP --> _00gR
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v591__00gR = np.sum(v590__00gP, axis = (2,)).reshape((1, 25600))

# op _00gS_power_combination_eval
# LANG: _00f_, _00fG --> _00gT
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v592__00gT = (v554__00fG)*(v564__00f_)
v592__00gT = v592__00gT.reshape((1, 25600))

# op _00gi_power_combination_eval
# LANG: _00gh --> _00gj
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v574__00gj = (v573__00gh**0.5)
v574__00gj = v574__00gj.reshape((1, 25600))

# op _00gq_single_tensor_sum_with_axis_eval
# LANG: _00gp --> _00gr
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v578__00gr = np.sum(v577__00gp, axis = (2,)).reshape((1, 25600))

# op _00gs_power_combination_eval
# LANG: _00fm, _00fG --> _00gt
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v579__00gt = (v544__00fm)*(v554__00fG)
v579__00gt = v579__00gt.reshape((1, 25600))

# op _00hd_power_combination_eval
# LANG: _00gd, _00fU --> _00he
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v603__00he = (v561__00fU)*(v571__00gd)
v603__00he = v603__00he.reshape((1, 25600, 3))

# op _008P_decompose_eval
# LANG: eel --> _008V, _008Q, _008R, _008U
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v328__008Q = ((v327_eel.flatten())[src_indices__008Q__008P]).reshape((1, 40, 4, 3))
v329__008R = ((v327_eel.flatten())[src_indices__008R__008P]).reshape((1, 40, 4, 3))
v331__008U = ((v327_eel.flatten())[src_indices__008U__008P]).reshape((1, 40, 4, 3))
v332__008V = ((v327_eel.flatten())[src_indices__008V__008P]).reshape((1, 40, 4, 3))

# op _00b6_power_combination_eval
# LANG: _00a_, _00b5 --> _00b7
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v406__00b7 = (v402__00a_)*(v405__00b5**-1)
v406__00b7 = v406__00b7.reshape((1, 44160))

# op _00bA_linear_combination_eval
# LANG: _00bp, _00bz --> _00bB
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v421__00bB = v415__00bp+v420__00bz

# op _00bg_power_combination_eval
# LANG: _00b9, _00bf --> _00bh
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v411__00bh = (v407__00b9)*(v410__00bf**-1)
v411__00bh = v411__00bh.reshape((1, 44160))

# op _00c3_power_combination_eval
# LANG: _00bX, _00c2 --> _00c4
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v436__00c4 = (v432__00bX)*(v435__00c2**-1)
v436__00c4 = v436__00c4.reshape((1, 44160))

# op _00cT_linear_combination_eval
# LANG: _00cQ, _00cO --> _00cU
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v462__00cU = v460__00cQ+-1*v459__00cO

# op _00cZ_power_combination_eval
# LANG: _00cY --> _00c_
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v465__00c_ = (v464__00cY**0.5)
v465__00c_ = v465__00c_.reshape((1, 44160))

# op _00cd_power_combination_eval
# LANG: _00c6, _00cc --> _00ce
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v441__00ce = (v437__00c6)*(v440__00cc**-1)
v441__00ce = v441__00ce.reshape((1, 44160))

# op _00cx_linear_combination_eval
# LANG: _00cm, _00cw --> _00cy
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v451__00cy = v445__00cm+v450__00cw

# op _00d2_linear_combination_eval
# LANG: _00cS, _00cO --> _00d3
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v467__00d3 = v461__00cS+-1*v459__00cO

# op _00d8_power_combination_eval
# LANG: _00d7 --> _00d9
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v470__00d9 = (v469__00d7**0.5)
v470__00d9 = v470__00d9.reshape((1, 44160))

# op _00dU_linear_combination_eval
# LANG: _00dT --> _00dV
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v494__00dV = _00dU_constant+v493__00dT

# op _00di_linear_combination_eval
# LANG: _00df, _00dh --> _00dj
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v475__00dj = v473__00df+-1*v474__00dh

# op _00ds_power_combination_eval
# LANG: _00dr --> _00dt
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v480__00dt = (v479__00dr)
v480__00dt = (v480__00dt*_00ds_coeff).reshape((1, 44160))

# op _00e3_linear_combination_eval
# LANG: _00e2 --> _00e4
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v499__00e4 = _00e3_constant+v498__00e2

# op _00eb_power_combination_eval
# LANG: _00dN, _00dP --> _00ec
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v503__00ec = (v490__00dN)*(v491__00dP)
v503__00ec = v503__00ec.reshape((1, 44160))

# op _00ed_power_combination_eval
# LANG: _00dL --> _00ee
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v504__00ee = (v489__00dL**2)
v504__00ee = v504__00ee.reshape((1, 44160))

# op _00en_linear_combination_eval
# LANG: _00ei, _00em --> _00eo
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v509__00eo = v506__00ei+-1*v508__00em

# op _00gA_power_combination_eval
# LANG: _00fG --> _00gB
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v583__00gB = (v554__00fG**-1)
v583__00gB = v583__00gB.reshape((1, 25600))

# op _00gU_linear_combination_eval
# LANG: _00gT, _00gR --> _00gV
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v593__00gV = v592__00gT+v591__00gR

# op _00gY_power_combination_eval
# LANG: _00fG --> _00gZ
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v595__00gZ = (v554__00fG**-1)
v595__00gZ = v595__00gZ.reshape((1, 25600))

# op _00g__power_combination_eval
# LANG: _00f_ --> _00h0
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v596__00h0 = (v564__00f_**-1)
v596__00h0 = v596__00h0.reshape((1, 25600))

# op _00gu_linear_combination_eval
# LANG: _00gt, _00gr --> _00gv
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v580__00gv = v579__00gt+v578__00gr

# op _00gy_power_combination_eval
# LANG: _00fm --> _00gz
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v582__00gz = (v544__00fm**-1)
v582__00gz = v582__00gz.reshape((1, 25600))

# op _00hD_power_combination_eval
# LANG: _00gd, _00fg --> _00hE
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v616__00hE = (v571__00gd)*(v541__00fg)
v616__00hE = v616__00hE.reshape((1, 25600, 3))

# op _00hf_single_tensor_sum_with_axis_eval
# LANG: _00he --> _00hg
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v604__00hg = np.sum(v603__00he, axis = (2,)).reshape((1, 25600))

# op _00hh_power_combination_eval
# LANG: _00gj, _00f_ --> _00hi
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v605__00hi = (v564__00f_)*(v574__00gj)
v605__00hi = v605__00hi.reshape((1, 25600))

# op _008S_linear_combination_eval
# LANG: _008Q, _008R --> _008T
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v330__008T = v328__008Q+-1*v329__008R

# op _008W_linear_combination_eval
# LANG: _008U, _008V --> _008X
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v333__008X = v331__008U+-1*v332__008V

# op _00bC_linear_combination_eval
# LANG: _00bB --> _00bD
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v422__00bD = _00bC_constant+v421__00bB

# op _00bi_linear_combination_eval
# LANG: _00b7, _00bh --> _00bj
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v412__00bj = v406__00b7+v411__00bh

# op _00cf_linear_combination_eval
# LANG: _00c4, _00ce --> _00cg
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v442__00cg = v436__00c4+v441__00ce

# op _00cz_linear_combination_eval
# LANG: _00cy --> _00cA
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v452__00cA = _00cz_constant+v451__00cy

# op _00d0_power_combination_eval
# LANG: _00cU, _00c_ --> _00d1
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v466__00d1 = (v462__00cU)*(v465__00c_**-1)
v466__00d1 = v466__00d1.reshape((1, 44160))

# op _00dQ_linear_combination_eval
# LANG: _00dN, _00dL --> _00dR
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v492__00dR = v490__00dN+-1*v489__00dL

# op _00dW_power_combination_eval
# LANG: _00dV --> _00dX
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v495__00dX = (v494__00dV**0.5)
v495__00dX = v495__00dX.reshape((1, 44160))

# op _00d__linear_combination_eval
# LANG: _00dP, _00dL --> _00e0
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v497__00e0 = v491__00dP+-1*v489__00dL

# op _00da_power_combination_eval
# LANG: _00d3, _00d9 --> _00db
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v471__00db = (v467__00d3)*(v470__00d9**-1)
v471__00db = v471__00db.reshape((1, 44160))

# op _00du_linear_combination_eval
# LANG: _00dj, _00dt --> _00dv
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v481__00dv = v475__00dj+v480__00dt

# op _00e5_power_combination_eval
# LANG: _00e4 --> _00e6
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v500__00e6 = (v499__00e4**0.5)
v500__00e6 = v500__00e6.reshape((1, 44160))

# op _00ef_linear_combination_eval
# LANG: _00ec, _00ee --> _00eg
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v505__00eg = v503__00ec+-1*v504__00ee

# op _00ep_power_combination_eval
# LANG: _00eo --> _00eq
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v510__00eq = (v509__00eo)
v510__00eq = (v510__00eq*_00ep_coeff).reshape((1, 44160))

# op _00gC_linear_combination_eval
# LANG: _00gz, _00gB --> _00gD
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v584__00gD = v582__00gz+v583__00gB

# op _00gW_power_combination_eval
# LANG: _00gV --> _00gX
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v594__00gX = (v593__00gV**-1)
v594__00gX = v594__00gX.reshape((1, 25600))

# op _00gw_power_combination_eval
# LANG: _00gv --> _00gx
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v581__00gx = (v580__00gv**-1)
v581__00gx = v581__00gx.reshape((1, 25600))

# op _00h1_linear_combination_eval
# LANG: _00gZ, _00h0 --> _00h2
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v597__00h2 = v595__00gZ+v596__00h0

# op _00hF_single_tensor_sum_with_axis_eval
# LANG: _00hE --> _00hG
# SHAPES: (1, 25600, 3) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v617__00hG = np.sum(v616__00hE, axis = (2,)).reshape((1, 25600))

# op _00hH_power_combination_eval
# LANG: _00fm, _00gj --> _00hI
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v618__00hI = (v574__00gj)*(v544__00fm)
v618__00hI = v618__00hI.reshape((1, 25600))

# op _00hj_linear_combination_eval
# LANG: _00hi, _00hg --> _00hk
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v606__00hk = v605__00hi+v604__00hg

# op _00hn_power_combination_eval
# LANG: _00f_ --> _00ho
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v608__00ho = (v564__00f_**-1)
v608__00ho = v608__00ho.reshape((1, 25600))

# op _00hp_power_combination_eval
# LANG: _00gj --> _00hq
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v609__00hq = (v574__00gj**-1)
v609__00hq = v609__00hq.reshape((1, 25600))

# op _008Y cross_product_eval
# LANG: _008T, _008X --> _008Z
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v334__008Z = np.cross(v330__008T, v333__008X, axisa = 3, axisb = 3, axisc = 3)

# op _00aN cross_product_eval
# LANG: _009J, _00a2 --> _00aO
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v396__00aO = np.cross(v362__009J, v372__00a2, axisa = 2, axisb = 2, axisc = 2)

# op _00bE_power_combination_eval
# LANG: _00bj, _00bD --> _00bF
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v423__00bF = (v412__00bj)*(v422__00bD**-1)
v423__00bF = v423__00bF.reshape((1, 44160))

# op _00bK cross_product_eval
# LANG: _00am, _00a2 --> _00bL
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v426__00bL = np.cross(v372__00a2, v382__00am, axisa = 2, axisb = 2, axisc = 2)

# op _00cB_power_combination_eval
# LANG: _00cg, _00cA --> _00cC
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v453__00cC = (v442__00cg)*(v452__00cA**-1)
v453__00cC = v453__00cC.reshape((1, 44160))

# op _00dY_power_combination_eval
# LANG: _00dR, _00dX --> _00dZ
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v496__00dZ = (v492__00dR)*(v495__00dX**-1)
v496__00dZ = v496__00dZ.reshape((1, 44160))

# op _00dc_linear_combination_eval
# LANG: _00d1, _00db --> _00dd
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v472__00dd = v466__00d1+v471__00db

# op _00dw_linear_combination_eval
# LANG: _00dv --> _00dx
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v482__00dx = _00dw_constant+v481__00dv

# op _00e7_power_combination_eval
# LANG: _00e0, _00e6 --> _00e8
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v501__00e8 = (v497__00e0)*(v500__00e6**-1)
v501__00e8 = v501__00e8.reshape((1, 44160))

# op _00er_linear_combination_eval
# LANG: _00eg, _00eq --> _00es
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v511__00es = v505__00eg+v510__00eq

# op _00gE_power_combination_eval
# LANG: _00gx, _00gD --> _00gF
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v585__00gF = (v581__00gx)*(v584__00gD)
v585__00gF = v585__00gF.reshape((1, 25600))

# op _00gK cross_product_eval
# LANG: _00fU, _00fA --> _00gL
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v588__00gL = np.cross(v551__00fA, v561__00fU, axisa = 2, axisb = 2, axisc = 2)

# op _00gk cross_product_eval
# LANG: _00fg, _00fA --> _00gl
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v575__00gl = np.cross(v541__00fg, v551__00fA, axisa = 2, axisb = 2, axisc = 2)

# op _00h3_power_combination_eval
# LANG: _00gX, _00h2 --> _00h4
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v598__00h4 = (v594__00gX)*(v597__00h2)
v598__00h4 = v598__00h4.reshape((1, 25600))

# op _00hJ_linear_combination_eval
# LANG: _00hI, _00hG --> _00hK
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v619__00hK = v618__00hI+v617__00hG

# op _00hN_power_combination_eval
# LANG: _00gj --> _00hO
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v621__00hO = (v574__00gj**-1)
v621__00hO = v621__00hO.reshape((1, 25600))

# op _00hP_power_combination_eval
# LANG: _00fm --> _00hQ
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v622__00hQ = (v544__00fm**-1)
v622__00hQ = v622__00hQ.reshape((1, 25600))

# op _00hl_power_combination_eval
# LANG: _00hk --> _00hm
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v607__00hm = (v606__00hk**-1)
v607__00hm = v607__00hm.reshape((1, 25600))

# op _00hr_linear_combination_eval
# LANG: _00ho, _00hq --> _00hs
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v610__00hs = v608__00ho+v609__00hq

# op _008__power_combination_eval
# LANG: _008Z --> _0090
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v335__0090 = (v334__008Z**2)
v335__0090 = v335__0090.reshape((1, 40, 4, 3))

# op _008q_indexed_passthrough_eval
# LANG: p, q, r --> ang_vel
# SHAPES: (1, 1), (1, 1), (1, 1) --> (1, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v313_ang_vel__temp[i_v310_p__008q_indexed_passthrough_eval] = v310_p.flatten()
v313_ang_vel = v313_ang_vel__temp.copy()
v313_ang_vel__temp[i_v311_q__008q_indexed_passthrough_eval] = v311_q.flatten()
v313_ang_vel = v313_ang_vel__temp.copy()
v313_ang_vel__temp[i_v312_r__008q_indexed_passthrough_eval] = v312_r.flatten()
v313_ang_vel = v313_ang_vel__temp.copy()

# op _008t expand_array_eval
# LANG: eel_rot_ref --> _008u
# SHAPES: (1, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v316__008u = np.einsum('ad,bc->abcd', v315_eel_rot_ref.reshape((1, 3)) ,np.ones((40, 4))).reshape((1, 40, 4, 3))

# op _00aP_power_combination_eval
# LANG: _00aO --> _00aQ
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v397__00aQ = (v396__00aO)
v397__00aQ = (v397__00aQ*_00aP_coeff).reshape((1, 44160, 3))

# op _00bG expand_array_eval
# LANG: _00bF --> _00bH
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v424__00bH = np.einsum('ab,c->abc', v423__00bF.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00bM_power_combination_eval
# LANG: _00bL --> _00bN
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v427__00bN = (v426__00bL)
v427__00bN = (v427__00bN*_00bM_coeff).reshape((1, 44160, 3))

# op _00cD expand_array_eval
# LANG: _00cC --> _00cE
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v454__00cE = np.einsum('ab,c->abc', v453__00cC.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00cH cross_product_eval
# LANG: _00aG, _00am --> _00cI
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v456__00cI = np.cross(v382__00am, v392__00aG, axisa = 2, axisb = 2, axisc = 2)

# op _00dy_power_combination_eval
# LANG: _00dd, _00dx --> _00dz
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v483__00dz = (v472__00dd)*(v482__00dx**-1)
v483__00dz = v483__00dz.reshape((1, 44160))

# op _00e9_linear_combination_eval
# LANG: _00dZ, _00e8 --> _00ea
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v502__00ea = v496__00dZ+v501__00e8

# op _00et_linear_combination_eval
# LANG: _00es --> _00eu
# SHAPES: (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v512__00eu = _00et_constant+v511__00es

# op _00gG expand_array_eval
# LANG: _00gF --> _00gH
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v586__00gH = np.einsum('ab,c->abc', v585__00gF.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00gM_power_combination_eval
# LANG: _00gL --> _00gN
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v589__00gN = (v588__00gL)
v589__00gN = (v589__00gN*_00gM_coeff).reshape((1, 25600, 3))

# op _00gm_power_combination_eval
# LANG: _00gl --> _00gn
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v576__00gn = (v575__00gl)
v576__00gn = (v576__00gn*_00gm_coeff).reshape((1, 25600, 3))

# op _00h5 expand_array_eval
# LANG: _00h4 --> _00h6
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v599__00h6 = np.einsum('ab,c->abc', v598__00h4.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00h9 cross_product_eval
# LANG: _00gd, _00fU --> _00ha
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v601__00ha = np.cross(v561__00fU, v571__00gd, axisa = 2, axisb = 2, axisc = 2)

# op _00hL_power_combination_eval
# LANG: _00hK --> _00hM
# SHAPES: (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v620__00hM = (v619__00hK**-1)
v620__00hM = v620__00hM.reshape((1, 25600))

# op _00hR_linear_combination_eval
# LANG: _00hO, _00hQ --> _00hS
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v623__00hS = v621__00hO+v622__00hQ

# op _00ht_power_combination_eval
# LANG: _00hm, _00hs --> _00hu
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v611__00hu = (v607__00hm)*(v610__00hs)
v611__00hu = v611__00hu.reshape((1, 25600))

# op _008v_linear_combination_eval
# LANG: _008u, eel_coll_pts_coords --> _008w
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v317__008w = v529_eel_coll_pts_coords+-1*v316__008u

# op _008x expand_array_eval
# LANG: ang_vel --> _008y
# SHAPES: (1, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v318__008y = np.einsum('ad,bc->abcd', v313_ang_vel.reshape((1, 3)) ,np.ones((40, 4))).reshape((1, 40, 4, 3))

# op _0091_single_tensor_sum_with_axis_eval
# LANG: _0090 --> _0092
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v336__0092 = np.sum(v335__0090, axis = (3,)).reshape((1, 40, 4))

# op _00bI_power_combination_eval
# LANG: _00bH, _00aQ --> _00bJ
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v425__00bJ = (v424__00bH)*(v397__00aQ)
v425__00bJ = v425__00bJ.reshape((1, 44160, 3))

# op _00cF_power_combination_eval
# LANG: _00cE, _00bN --> _00cG
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v455__00cG = (v454__00cE)*(v427__00bN)
v455__00cG = v455__00cG.reshape((1, 44160, 3))

# op _00cJ_power_combination_eval
# LANG: _00cI --> _00cK
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v457__00cK = (v456__00cI)
v457__00cK = (v457__00cK*_00cJ_coeff).reshape((1, 44160, 3))

# op _00dA expand_array_eval
# LANG: _00dz --> _00dB
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v484__00dB = np.einsum('ab,c->abc', v483__00dz.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00dE cross_product_eval
# LANG: _00aG, _009J --> _00dF
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v486__00dF = np.cross(v392__00aG, v362__009J, axisa = 2, axisb = 2, axisc = 2)

# op _00ev_power_combination_eval
# LANG: _00ea, _00eu --> _00ew
# SHAPES: (1, 44160), (1, 44160) --> (1, 44160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v513__00ew = (v502__00ea)*(v512__00eu**-1)
v513__00ew = v513__00ew.reshape((1, 44160))

# op _00gI_power_combination_eval
# LANG: _00gH, _00gn --> _00gJ
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v587__00gJ = (v586__00gH)*(v576__00gn)
v587__00gJ = v587__00gJ.reshape((1, 25600, 3))

# op _00h7_power_combination_eval
# LANG: _00h6, _00gN --> _00h8
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v600__00h8 = (v599__00h6)*(v589__00gN)
v600__00h8 = v600__00h8.reshape((1, 25600, 3))

# op _00hT_power_combination_eval
# LANG: _00hM, _00hS --> _00hU
# SHAPES: (1, 25600), (1, 25600) --> (1, 25600)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v624__00hU = (v620__00hM)*(v623__00hS)
v624__00hU = v624__00hU.reshape((1, 25600))

# op _00hb_power_combination_eval
# LANG: _00ha --> _00hc
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v602__00hc = (v601__00ha)
v602__00hc = (v602__00hc*_00hb_coeff).reshape((1, 25600, 3))

# op _00hv expand_array_eval
# LANG: _00hu --> _00hw
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v612__00hw = np.einsum('ab,c->abc', v611__00hu.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00hz cross_product_eval
# LANG: _00gd, _00fg --> _00hA
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v614__00hA = np.cross(v571__00gd, v541__00fg, axisa = 2, axisb = 2, axisc = 2)

# op _008z cross_product_eval
# LANG: _008y, _008w --> _008A
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v319__008A = np.cross(v318__008y, v317__008w, axisa = 3, axisb = 3, axisc = 3)

# op _0093_power_combination_eval
# LANG: _0092 --> _0094
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v337__0094 = (v336__0092**0.5)
v337__0094 = v337__0094.reshape((1, 40, 4))

# op _00dC_power_combination_eval
# LANG: _00dB, _00cK --> _00dD
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v485__00dD = (v484__00dB)*(v457__00cK)
v485__00dD = v485__00dD.reshape((1, 44160, 3))

# op _00dG_power_combination_eval
# LANG: _00dF --> _00dH
# SHAPES: (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v487__00dH = (v486__00dF)
v487__00dH = (v487__00dH*_00dG_coeff).reshape((1, 44160, 3))

# op _00eB_linear_combination_eval
# LANG: _00bJ, _00cG --> _00eC
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v516__00eC = v425__00bJ+v455__00cG

# op _00ex expand_array_eval
# LANG: _00ew --> _00ey
# SHAPES: (1, 44160) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v514__00ey = np.einsum('ab,c->abc', v513__00ew.reshape((1, 44160)) ,np.ones((3,))).reshape((1, 44160, 3))

# op _00hB_power_combination_eval
# LANG: _00hA --> _00hC
# SHAPES: (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v615__00hC = (v614__00hA)
v615__00hC = (v615__00hC*_00hB_coeff).reshape((1, 25600, 3))

# op _00hV expand_array_eval
# LANG: _00hU --> _00hW
# SHAPES: (1, 25600) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v625__00hW = np.einsum('ab,c->abc', v624__00hU.reshape((1, 25600)) ,np.ones((3,))).reshape((1, 25600, 3))

# op _00hZ_linear_combination_eval
# LANG: _00gJ, _00h8 --> _00h_
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v627__00h_ = v587__00gJ+v600__00h8

# op _00hx_power_combination_eval
# LANG: _00hw, _00hc --> _00hy
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v613__00hy = (v612__00hw)*(v602__00hc)
v613__00hy = v613__00hy.reshape((1, 25600, 3))

# op _008B reshape_eval
# LANG: _008A --> _008C
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v320__008C = v319__008A.reshape((1, 160, 3))

# op _008D expand_array_eval
# LANG: frame_vel --> _008E
# SHAPES: (1, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v321__008E = np.einsum('ac,b->abc', v639_frame_vel.reshape((1, 3)) ,np.ones((160,))).reshape((1, 160, 3))

# op _0095 expand_array_eval
# LANG: _0094 --> _0096
# SHAPES: (1, 40, 4) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v338__0096 = np.einsum('abc,d->abcd', v337__0094.reshape((1, 40, 4)) ,np.ones((3,))).reshape((1, 40, 4, 3))

# op _00eD_linear_combination_eval
# LANG: _00eC, _00dD --> _00eE
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v517__00eE = v516__00eC+v485__00dD

# op _00ez_power_combination_eval
# LANG: _00ey, _00dH --> _00eA
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v515__00eA = (v514__00ey)*(v487__00dH)
v515__00eA = v515__00eA.reshape((1, 44160, 3))

# op _00hX_power_combination_eval
# LANG: _00hW, _00hC --> _00hY
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v626__00hY = (v625__00hW)*(v615__00hC)
v626__00hY = v626__00hY.reshape((1, 25600, 3))

# op _00i0_linear_combination_eval
# LANG: _00h_, _00hy --> _00i1
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v628__00i1 = v627__00h_+v613__00hy

# op _008G_linear_combination_eval
# LANG: _008C, _008E --> _008H
# SHAPES: (1, 160, 3), (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v323__008H = v320__008C+v321__008E

# op _008I reshape_eval
# LANG: eel_coll_vel --> _008J
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v324__008J = v322_eel_coll_vel.reshape((1, 160, 3))

# op _0097_power_combination_eval
# LANG: _008Z, _0096 --> eel_bd_vtx_normals
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.ComputeNormal
v632_eel_bd_vtx_normals = (v334__008Z)*(v338__0096**-1)
v632_eel_bd_vtx_normals = v632_eel_bd_vtx_normals.reshape((1, 40, 4, 3))

# op _00eF_linear_combination_eval
# LANG: _00eE, _00eA --> aic_M00
# SHAPES: (1, 44160, 3), (1, 44160, 3) --> (1, 44160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic.aic_bd_w_seperate
v518_aic_M00 = v517__00eE+v515__00eA

# op _00i2_linear_combination_eval
# LANG: _00i1, _00hY --> aic_bd00
# SHAPES: (1, 25600, 3), (1, 25600, 3) --> (1, 25600, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd.aic_bd_w_seperate
v629_aic_bd00 = v628__00i1+v626__00hY

# op _008K_linear_combination_eval
# LANG: _008H, _008J --> _008L
# SHAPES: (1, 160, 3), (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v325__008L = v323__008H+v324__008J

# op _009m reshape_eval
# LANG: aic_M00 --> _009n
# SHAPES: (1, 44160, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v349__009n = v518_aic_M00.reshape((1, 160, 276, 3))

# op _00eK reshape_eval
# LANG: eel_bd_vtx_normals --> _00eL
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v522__00eL = v632_eel_bd_vtx_normals.reshape((1, 160, 3))

# op _00eU reshape_eval
# LANG: aic_bd00 --> _00eV
# SHAPES: (1, 25600, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v528__00eV = v629_aic_bd00.reshape((1, 160, 160, 3))

# op _00i7 reshape_eval
# LANG: eel_bd_vtx_normals --> _00i8
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v633__00i8 = v632_eel_bd_vtx_normals.reshape((1, 160, 3))

# op _008M_linear_combination_eval
# LANG: _008L --> eel_kinematic_vel
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.KinematicVelocityComp
v341_eel_kinematic_vel = -1*v325__008L

# op _009c reshape_eval
# LANG: eel_bd_vtx_normals --> _009d
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v343__009d = v632_eel_bd_vtx_normals.reshape((1, 160, 3))

# op _009o_indexed_passthrough_eval
# LANG: _009n --> aic_M
# SHAPES: (1, 160, 276, 3) --> (1, 160, 276, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic
v520_aic_M__temp[i_v349__009n__009o_indexed_passthrough_eval] = v349__009n.flatten()
v520_aic_M = v520_aic_M__temp.copy()

# op _00eM_indexed_passthrough_eval
# LANG: _00eL --> normal_concatenated_M_mat
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
v519_normal_concatenated_M_mat__temp[i_v522__00eL__00eM_indexed_passthrough_eval] = v522__00eL.flatten()
v519_normal_concatenated_M_mat = v519_normal_concatenated_M_mat__temp.copy()

# op _00eW_indexed_passthrough_eval
# LANG: _00eV --> aic_bd
# SHAPES: (1, 160, 160, 3) --> (1, 160, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.AssembleAic_bd
v631_aic_bd__temp[i_v528__00eV__00eW_indexed_passthrough_eval] = v528__00eV.flatten()
v631_aic_bd = v631_aic_bd__temp.copy()

# op _00i9_indexed_passthrough_eval
# LANG: _00i8 --> normal_concatenated_aic_bd_proj
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
v630_normal_concatenated_aic_bd_proj__temp[i_v633__00i8__00i9_indexed_passthrough_eval] = v633__00i8.flatten()
v630_normal_concatenated_aic_bd_proj = v630_normal_concatenated_aic_bd_proj__temp.copy()

# op _009e_custom_explicit_eval
# LANG: _009d, eel_kinematic_vel --> b
# SHAPES: (1, 160, 3), (1, 160, 3) --> (1, 160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
temp = _009e_custom_explicit_func_b.solve(v341_eel_kinematic_vel, v343__009d)
v344_b = temp[0].copy()

# op _00eN_custom_explicit_eval
# LANG: normal_concatenated_M_mat, aic_M --> M_mat
# SHAPES: (1, 160, 3), (1, 160, 276, 3) --> (1, 160, 276)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic
temp = _00eN_custom_explicit_func_M_mat.solve(v520_aic_M, v519_normal_concatenated_M_mat)
v523_M_mat = temp[0].copy()

# op _00ia_custom_explicit_eval
# LANG: normal_concatenated_aic_bd_proj, aic_bd --> aic_bd_proj
# SHAPES: (1, 160, 3), (1, 160, 160, 3) --> (1, 160, 160)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_aic_bd
temp = _00ia_custom_explicit_func_aic_bd_proj.solve(v631_aic_bd, v630_normal_concatenated_aic_bd_proj)
v634_aic_bd_proj = temp[0].copy()

# op _007G_indexed_passthrough_eval
# LANG: eel_gamma_w --> gamma_w
# SHAPES: (1, 69, 4) --> (1, 69, 4)
# full namespace: combine_gamma_w
v297_gamma_w__temp[i_v285_eel_gamma_w__007G_indexed_passthrough_eval] = v285_eel_gamma_w.flatten()
v297_gamma_w = v297_gamma_w__temp.copy()

# op _0081_newton_implict_eval
# LANG: b, M_mat, aic_bd_proj, gamma_w --> gamma_b
# SHAPES: (1, 160), (1, 160, 276), (1, 160, 160), (1, 69, 4) --> (1, 160)
# full namespace: solve_gamma_b_group
_0081_newton.set_guess(initial_guess_v635_gamma_b)
_0081_newton_out = _0081_newton.solve(v634_aic_bd_proj, v523_M_mat, v297_gamma_w, v344_b)
v635_gamma_b = _0081_newton_out[0]

# op _00ik_linear_combination_eval
# LANG: frame_vel --> _00il
# SHAPES: (1, 3) --> (1, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v640__00il = -1*v639_frame_vel

# op _00im expand_array_eval
# LANG: _00il --> eel_wake_kinematic_vel
# SHAPES: (1, 3) --> (1, 69, 5, 3)
# full namespace: ComputeWakeTotalVel.ComputeWakeKinematicVel
v641_eel_wake_kinematic_vel = np.einsum('ad,bc->abcd', v640__00il.reshape((1, 3)) ,np.ones((69, 5))).reshape((1, 69, 5, 3))

# op _00ih_linear_combination_eval
# LANG: eel_wake_kinematic_vel --> eel_wake_total_vel
# SHAPES: (1, 69, 5, 3) --> (1, 69, 5, 3)
# full namespace: ComputeWakeTotalVel
v638_eel_wake_total_vel = v641_eel_wake_kinematic_vel

# op _003R_decompose_eval
# LANG: eel_wake_total_vel --> _004z, _003S
# SHAPES: (1, 69, 5, 3) --> (1, 68, 5, 3), (1, 1, 5, 3)
# full namespace: 
v147__003S = ((v638_eel_wake_total_vel.flatten())[src_indices__003S__003R]).reshape((1, 1, 5, 3))
v170__004z = ((v638_eel_wake_total_vel.flatten())[src_indices__004z__003R]).reshape((1, 68, 5, 3))

# op _004h_power_combination_eval
# LANG: _003S --> _004i
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v160__004i = (v147__003S)
v160__004i = (v160__004i*_004h_coeff).reshape((1, 1, 5, 3))

# op _003P_decompose_eval
# LANG: eel_bd_vtx_coords --> _003Q
# SHAPES: (1, 41, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v146__003Q = ((v530_eel_bd_vtx_coords.flatten())[src_indices__003Q__003P]).reshape((1, 1, 5, 3))

# op _004j_power_combination_eval
# LANG: _004i --> _004k
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v161__004k = (v160__004i)
v161__004k = (v161__004k*_004j_coeff).reshape((1, 1, 5, 3))

# op _003Z_decompose_eval
# LANG: eel_wake_coords --> _004w, _003_, _004t
# SHAPES: (1, 69, 5, 3) --> (1, 68, 5, 3), (1, 1, 5, 3), (1, 68, 5, 3)
# full namespace: 
v151__003_ = ((v303_eel_wake_coords.flatten())[src_indices__003___003Z]).reshape((1, 1, 5, 3))
v166__004t = ((v303_eel_wake_coords.flatten())[src_indices__004t__003Z]).reshape((1, 68, 5, 3))
v168__004w = ((v303_eel_wake_coords.flatten())[src_indices__004w__003Z]).reshape((1, 68, 5, 3))

# op _004l_linear_combination_eval
# LANG: _003Q, _004k --> _004m
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v162__004m = v146__003Q+v161__004k

# op _0067_linear_combination_eval
# LANG: _005r, _005u --> _0068
# SHAPES: (1, 40, 5, 3), (1, 40, 5, 3) --> (1, 40, 5, 3)
# full namespace: MeshPreprocessing_comp
v230__0068 = v204__005r+-1*v206__005u

# op _004Q_power_combination_eval
# LANG: u --> _004R
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v184__004R = (v174_u**2)
v184__004R = v184__004R.reshape((1, 1))

# op _004S_power_combination_eval
# LANG: v --> _004T
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v185__004T = (v175_v**2)
v185__004T = v185__004T.reshape((1, 1))

# op _004n_linear_combination_eval
# LANG: _004m, _003_ --> _004o
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v163__004o = v162__004m+-1*v151__003_

# op _005n_linear_combination_eval
# LANG: eel --> _005o
# SHAPES: (1, 41, 5, 3) --> (1, 41, 5, 3)
# full namespace: MeshPreprocessing_comp
v202__005o = v327_eel

# op _0069 pnorm_axis_eval
# LANG: _0068 --> _006a
# SHAPES: (1, 40, 5, 3) --> (1, 40, 5)
# full namespace: MeshPreprocessing_comp
v231__006a = np.sum(v230__0068**2,axis=(3,))**(1 / 2)

# op _004U_linear_combination_eval
# LANG: _004R, _004T --> _004V
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v186__004V = v184__004R+v185__004T

# op _004W_power_combination_eval
# LANG: w --> _004X
# SHAPES: (1, 1) --> (1, 1)
# full namespace: adapter_comp
v187__004X = (v210_w**2)
v187__004X = v187__004X.reshape((1, 1))

# op _004p reshape_eval
# LANG: _004o --> _004q
# SHAPES: (1, 1, 5, 3) --> (1, 5, 3)
# full namespace: 
v164__004q = v163__004o.reshape((1, 5, 3))

# op _006Q_power_combination_eval
# LANG: _006P --> _006R
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v257__006R = (v256__006P)
v257__006R = (v257__006R*_006Q_coeff).reshape((1, 40, 4, 3))

# op _006T_power_combination_eval
# LANG: _006S --> _006U
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v259__006U = (v258__006S)
v259__006U = (v259__006U*_006T_coeff).reshape((1, 40, 4, 3))

# op _006b_decompose_eval
# LANG: _006a --> _006d, _006c
# SHAPES: (1, 40, 5) --> (1, 40, 4), (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v232__006c = ((v231__006a.flatten())[src_indices__006c__006b]).reshape((1, 40, 4))
v233__006d = ((v231__006a.flatten())[src_indices__006d__006b]).reshape((1, 40, 4))

# op _006v_decompose_eval
# LANG: _005o --> _006B, _006w, _006x, _006A
# SHAPES: (1, 41, 5, 3) --> (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3), (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v244__006w = ((v202__005o.flatten())[src_indices__006w__006v]).reshape((1, 40, 4, 3))
v245__006x = ((v202__005o.flatten())[src_indices__006x__006v]).reshape((1, 40, 4, 3))
v247__006A = ((v202__005o.flatten())[src_indices__006A__006v]).reshape((1, 40, 4, 3))
v248__006B = ((v202__005o.flatten())[src_indices__006B__006v]).reshape((1, 40, 4, 3))

# op _003T_power_combination_eval
# LANG: _003S --> _003U
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v148__003U = (v147__003S)
v148__003U = (v148__003U*_003T_coeff).reshape((1, 1, 5, 3))

# op _004Y_linear_combination_eval
# LANG: _004V, _004X --> v_inf_sq
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v276_v_inf_sq = v186__004V+v187__004X

# op _004r expand_array_eval
# LANG: _004q --> _004s
# SHAPES: (1, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v165__004s = np.einsum('acd,b->abcd', v164__004q.reshape((1, 5, 3)) ,np.ones((68,))).reshape((1, 68, 5, 3))

# op _006C_linear_combination_eval
# LANG: _006A, _006B --> _006D
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v249__006D = v247__006A+-1*v248__006B

# op _006V_linear_combination_eval
# LANG: _006R, _006U --> _006W
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v260__006W = v257__006R+v259__006U

# op _006Y_power_combination_eval
# LANG: _006X --> _006Z
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v262__006Z = (v261__006X)
v262__006Z = (v262__006Z*_006Y_coeff).reshape((1, 40, 4, 3))

# op _006e_linear_combination_eval
# LANG: _006c, _006d --> _006f
# SHAPES: (1, 40, 4), (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v234__006f = v232__006c+v233__006d

# op _006y_linear_combination_eval
# LANG: _006w, _006x --> _006z
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v246__006z = v244__006w+-1*v245__006x

# op _00id_decompose_eval
# LANG: gamma_b --> eel_gamma_b
# SHAPES: (1, 160) --> (1, 160)
# full namespace: seperate_gamma_b
v636_eel_gamma_b = ((v635_gamma_b.flatten())[src_indices_eel_gamma_b__00id]).reshape((1, 160))

# op _003V_power_combination_eval
# LANG: _003U --> _003W
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v149__003W = (v148__003U)
v149__003W = (v149__003W*_003V_coeff).reshape((1, 1, 5, 3))

# op _003s_decompose_eval
# LANG: eel_gamma_b --> _003t
# SHAPES: (1, 160) --> (1, 4)
# full namespace: 
v132__003t = ((v636_eel_gamma_b.flatten())[src_indices__003t__003s]).reshape((1, 4))

# op _004u_linear_combination_eval
# LANG: _004s, _004t --> _004v
# SHAPES: (1, 68, 5, 3), (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v167__004v = v165__004s+v166__004t

# op _006E cross_product_eval
# LANG: _006z, _006D --> _006F
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v250__006F = np.cross(v246__006z, v249__006D, axisa = 3, axisb = 3, axisc = 3)

# op _006__linear_combination_eval
# LANG: _006W, _006Z --> _0070
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v263__0070 = v260__006W+v262__006Z

# op _006g_power_combination_eval
# LANG: _006f --> eel_chord_length
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v235_eel_chord_length = (v234__006f)
v235_eel_chord_length = (v235_eel_chord_length*_006g_coeff).reshape((1, 40, 4))

# op _006k_linear_combination_eval
# LANG: _006i, _006j --> _006l
# SHAPES: (1, 41, 4, 3), (1, 41, 4, 3) --> (1, 41, 4, 3)
# full namespace: MeshPreprocessing_comp
v238__006l = v236__006i+-1*v237__006j

# op _0071_power_combination_eval
# LANG: _0060 --> _0072
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v264__0072 = (v226__0060)
v264__0072 = (v264__0072*_0071_coeff).reshape((1, 40, 4, 3))

# op _007u_power_combination_eval
# LANG: v_inf_sq --> _007v
# SHAPES: (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v279__007v = (v276_v_inf_sq**0.5)
v279__007v = v279__007v.reshape((1, 1))

# op _003X_linear_combination_eval
# LANG: _003Q, _003W --> _003Y
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v150__003Y = v146__003Q+v149__003W

# op _003u reshape_eval
# LANG: _003t --> _003v
# SHAPES: (1, 4) --> (1, 1, 4)
# full namespace: 
v133__003v = v132__003t.reshape((1, 1, 4))

# op _003w_decompose_eval
# LANG: eel_gamma_w --> _003E, _003x, _003D
# SHAPES: (1, 69, 4) --> (1, 68, 4), (1, 1, 4), (1, 68, 4)
# full namespace: 
v134__003x = ((v285_eel_gamma_w.flatten())[src_indices__003x__003w]).reshape((1, 1, 4))
v137__003D = ((v285_eel_gamma_w.flatten())[src_indices__003D__003w]).reshape((1, 68, 4))
v138__003E = ((v285_eel_gamma_w.flatten())[src_indices__003E__003w]).reshape((1, 68, 4))

# op _004A_power_combination_eval
# LANG: _004z --> _004B
# SHAPES: (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v171__004B = (v170__004z)
v171__004B = (v171__004B*_004A_coeff).reshape((1, 68, 5, 3))

# op _004x_linear_combination_eval
# LANG: _004v, _004w --> _004y
# SHAPES: (1, 68, 5, 3), (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v169__004y = v167__004v+-1*v168__004w

# op _006G_power_combination_eval
# LANG: _006F --> _006H
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v251__006H = (v250__006F**2)
v251__006H = v251__006H.reshape((1, 40, 4, 3))

# op _006m pnorm_axis_eval
# LANG: _006l --> _006n
# SHAPES: (1, 41, 4, 3) --> (1, 41, 4)
# full namespace: MeshPreprocessing_comp
v239__006n = np.sum(v238__006l**2,axis=(3,))**(1 / 2)

# op _0073_linear_combination_eval
# LANG: _0070, _0072 --> _0074
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v265__0074 = v263__0070+v264__0072

# op _007a_power_combination_eval
# LANG: _006P --> _007b
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v268__007b = (v256__006P)
v268__007b = (v268__007b*_007a_coeff).reshape((1, 40, 4, 3))

# op _007c_power_combination_eval
# LANG: _006X --> _007d
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v269__007d = (v261__006X)
v269__007d = (v269__007d*_007c_coeff).reshape((1, 40, 4, 3))

# op _007q_single_tensor_sum_with_axis_eval
# LANG: eel_chord_length --> _007r
# SHAPES: (1, 40, 4) --> (1, 4)
# full namespace: MeshPreprocessing_comp
v277__007r = np.sum(v235_eel_chord_length, axis = (1,)).reshape((1, 4))

# op _007w_power_combination_eval
# LANG: density, _007v --> _007x
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: MeshPreprocessing_comp
v280__007x = (v275_density)*(v279__007v)
v280__007x = v280__007x.reshape((1, 1))

# op _003F_linear_combination_eval
# LANG: _003D, _003E --> _003G
# SHAPES: (1, 68, 4), (1, 68, 4) --> (1, 68, 4)
# full namespace: 
v139__003G = v137__003D+-1*v138__003E

# op _003y_linear_combination_eval
# LANG: _003v, _003x --> _003z
# SHAPES: (1, 1, 4), (1, 1, 4) --> (1, 1, 4)
# full namespace: 
v135__003z = v133__003v+-1*v134__003x

# op _0040_linear_combination_eval
# LANG: _003Y, _003_ --> _0041
# SHAPES: (1, 1, 5, 3), (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v152__0041 = v150__003Y+-1*v151__003_

# op _004C_linear_combination_eval
# LANG: _004y, _004B --> _004D
# SHAPES: (1, 68, 5, 3), (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v172__004D = v169__004y+v171__004B

# op _006I_single_tensor_sum_with_axis_eval
# LANG: _006H --> _006J
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v252__006J = np.sum(v251__006H, axis = (3,)).reshape((1, 40, 4))

# op _006o_decompose_eval
# LANG: _006n --> _006q, _006p
# SHAPES: (1, 41, 4) --> (1, 40, 4), (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v240__006p = ((v239__006n.flatten())[src_indices__006p__006o]).reshape((1, 40, 4))
v241__006q = ((v239__006n.flatten())[src_indices__006q__006o]).reshape((1, 40, 4))

# op _0075 reshape_eval
# LANG: _0074 --> _0076
# SHAPES: (1, 40, 4, 3) --> (1, 160, 3)
# full namespace: MeshPreprocessing_comp
v266__0076 = v265__0074.reshape((1, 160, 3))

# op _007e_linear_combination_eval
# LANG: _007b, _007d --> _007f
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v270__007f = v268__007b+v269__007d

# op _007g_power_combination_eval
# LANG: _006S --> _007h
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v271__007h = (v258__006S)
v271__007h = (v271__007h*_007g_coeff).reshape((1, 40, 4, 3))

# op _007s reshape_eval
# LANG: _007r --> _007t
# SHAPES: (1, 4) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v278__007t = v277__007r.reshape((1, 4, 1))

# op _007y expand_array_eval
# LANG: _007x --> _007z
# SHAPES: (1, 1) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v281__007z = np.einsum('ac,b->abc', v280__007x.reshape((1, 1)) ,np.ones((4,))).reshape((1, 4, 1))

# op _003A_power_combination_eval
# LANG: _003z --> _003B
# SHAPES: (1, 1, 4) --> (1, 1, 4)
# full namespace: 
v136__003B = (v135__003z)
v136__003B = (v136__003B*_003A_coeff).reshape((1, 1, 4))

# op _003H_power_combination_eval
# LANG: _003G --> _003I
# SHAPES: (1, 68, 4) --> (1, 68, 4)
# full namespace: 
v140__003I = (v139__003G)
v140__003I = (v140__003I*_003H_coeff).reshape((1, 68, 4))

# op _0042_power_combination_eval
# LANG: _0041 --> _0043
# SHAPES: (1, 1, 5, 3) --> (1, 1, 5, 3)
# full namespace: 
v153__0043 = (v152__0041)
v153__0043 = (v153__0043*_0042_coeff).reshape((1, 1, 5, 3))

# op _004E_power_combination_eval
# LANG: _004D --> _004F
# SHAPES: (1, 68, 5, 3) --> (1, 68, 5, 3)
# full namespace: 
v173__004F = (v172__004D)
v173__004F = (v173__004F*_004E_coeff).reshape((1, 68, 5, 3))

# op _006K_power_combination_eval
# LANG: _006J --> _006L
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v253__006L = (v252__006J**0.5)
v253__006L = v253__006L.reshape((1, 40, 4))

# op _006r_linear_combination_eval
# LANG: _006p, _006q --> _006s
# SHAPES: (1, 40, 4), (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v242__006s = v240__006p+v241__006q

# op _0077_linear_combination_eval
# LANG: _0076 --> _0078
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: MeshPreprocessing_comp
v267__0078 = -1*v266__0076

# op _007A_power_combination_eval
# LANG: _007z, _007t --> _007B
# SHAPES: (1, 4, 1), (1, 4, 1) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v282__007B = (v281__007z)*(v278__007t)
v282__007B = v282__007B.reshape((1, 4, 1))

# op _007i_linear_combination_eval
# LANG: _007f, _007h --> _007j
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v272__007j = v270__007f+v271__007h

# op _007k_power_combination_eval
# LANG: _0060 --> _007l
# SHAPES: (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v273__007l = (v226__0060)
v273__007l = (v273__007l*_007k_coeff).reshape((1, 40, 4, 3))

# op _003C_indexed_passthrough_eval
# LANG: _003B, _003I --> eel_dgammaw_dt
# SHAPES: (1, 1, 4), (1, 68, 4) --> (1, 69, 4)
# full namespace: 
v131_eel_dgammaw_dt__temp[i_v136__003B__003C_indexed_passthrough_eval] = v136__003B.flatten()
v131_eel_dgammaw_dt = v131_eel_dgammaw_dt__temp.copy()
v131_eel_dgammaw_dt__temp[i_v140__003I__003C_indexed_passthrough_eval] = v140__003I.flatten()
v131_eel_dgammaw_dt = v131_eel_dgammaw_dt__temp.copy()

# op _0044_indexed_passthrough_eval
# LANG: _0043, _004F --> eel_dwake_coords_dt
# SHAPES: (1, 1, 5, 3), (1, 68, 5, 3) --> (1, 69, 5, 3)
# full namespace: 
v145_eel_dwake_coords_dt__temp[i_v153__0043__0044_indexed_passthrough_eval] = v153__0043.flatten()
v145_eel_dwake_coords_dt = v145_eel_dwake_coords_dt__temp.copy()
v145_eel_dwake_coords_dt__temp[i_v173__004F__0044_indexed_passthrough_eval] = v173__004F.flatten()
v145_eel_dwake_coords_dt = v145_eel_dwake_coords_dt__temp.copy()

# op _005b_linear_combination_eval
# LANG: theta, gamma --> alpha
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v195_alpha = v180_theta+-1*v182_gamma

# op _005d_linear_combination_eval
# LANG: psi, psiw --> beta
# SHAPES: (1, 1), (1, 1) --> (1, 1)
# full namespace: adapter_comp
v196_beta = v181_psi+v183_psiw

# op _006M_power_combination_eval
# LANG: _006L --> eel_s_panel
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v254_eel_s_panel = (v253__006L)
v254_eel_s_panel = (v254_eel_s_panel*_006M_coeff).reshape((1, 40, 4))

# op _006t_power_combination_eval
# LANG: _006s --> eel_span_length
# SHAPES: (1, 40, 4) --> (1, 40, 4)
# full namespace: MeshPreprocessing_comp
v243_eel_span_length = (v242__006s)
v243_eel_span_length = (v243_eel_span_length*_006t_coeff).reshape((1, 40, 4))

# op _0079_indexed_passthrough_eval
# LANG: _0078 --> bd_vec
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: MeshPreprocessing_comp
v255_bd_vec__temp[i_v267__0078__0079_indexed_passthrough_eval] = v267__0078.flatten()
v255_bd_vec = v255_bd_vec__temp.copy()

# op _007C_power_combination_eval
# LANG: _007B --> eel_re_span
# SHAPES: (1, 4, 1) --> (1, 4, 1)
# full namespace: MeshPreprocessing_comp
v283_eel_re_span = (v282__007B)
v283_eel_re_span = (v283_eel_re_span*_007C_coeff).reshape((1, 4, 1))

# op _007m_linear_combination_eval
# LANG: _007j, _007l --> eel_eval_pts_coords
# SHAPES: (1, 40, 4, 3), (1, 40, 4, 3) --> (1, 40, 4, 3)
# full namespace: MeshPreprocessing_comp
v274_eel_eval_pts_coords = v272__007j+v273__007l

# op _009g_indexed_passthrough_eval
# LANG: _009d --> normal_concatenated_b
# SHAPES: (1, 160, 3) --> (1, 160, 3)
# full namespace: solve_gamma_b_group.prepossing_before_Solve.RHS_group.Projection_k_vel
v340_normal_concatenated_b__temp[i_v343__009d__009g_indexed_passthrough_eval] = v343__009d.flatten()
v340_normal_concatenated_b = v340_normal_concatenated_b__temp.copy()