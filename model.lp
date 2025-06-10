\ Model ev_city
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x525a512137956280
Maximize
  0.4851502024813346 act_current_ev_dis[0,0,0]
   + 0.4934120848334382 act_current_ev_dis[0,0,1]
   + 0.4710956494083977 act_current_ev_dis[0,0,2]
   + 0.4052998889711173 act_current_ev_dis[0,0,3]
   + 0.343582068955177 act_current_ev_dis[0,0,4]
   + 0.3100232381483685 act_current_ev_dis[0,0,5]
   + 0.2930255843432106 act_current_ev_dis[0,0,6]
   + 0.3167449809223818 act_current_ev_dis[0,0,7]
   + 0.3160840303342135 act_current_ev_dis[0,0,8]
   + 0.3188151280475881 act_current_ev_dis[0,0,9]
   + 0.4479312019079715 act_current_ev_dis[0,0,10]
   + 0.4222975427762752 act_current_ev_dis[0,0,11]
   + 0.3705750415606534 act_current_ev_dis[0,0,12]
   + 0.4159374522108823 act_current_ev_dis[0,0,13]
   + 0.2968354032995391 act_current_ev_dis[0,0,14]
   + 0.3809133064208705 act_current_ev_dis[0,0,15]
   + 0.3468057619182243 act_current_ev_dis[0,0,16]
   + 0.3120123252957806 act_current_ev_dis[0,0,17]
   + 0.3173061653840341 act_current_ev_dis[0,0,18]
   + 0.3357504280236736 act_current_ev_dis[0,0,19]
   + 0.2924955767960944 act_current_ev_dis[0,0,20]
   + 0.3378517520634162 act_current_ev_dis[0,0,21]
   + 0.3295898697113126 act_current_ev_dis[0,0,22]
   + 0.40529365358821 act_current_ev_dis[0,0,23]
   - 0.4851502024813346 act_current_ev_ch[0,0,0]
   - 0.4934120848334382 act_current_ev_ch[0,0,1]
   - 0.4710956494083977 act_current_ev_ch[0,0,2]
   - 0.4052998889711173 act_current_ev_ch[0,0,3]
   - 0.343582068955177 act_current_ev_ch[0,0,4]
   - 0.3100232381483685 act_current_ev_ch[0,0,5]
   - 0.2930255843432106 act_current_ev_ch[0,0,6]
   - 0.3167449809223818 act_current_ev_ch[0,0,7]
   - 0.3160840303342135 act_current_ev_ch[0,0,8]
   - 0.3188151280475881 act_current_ev_ch[0,0,9]
   - 0.4479312019079715 act_current_ev_ch[0,0,10]
   - 0.4222975427762752 act_current_ev_ch[0,0,11]
   - 0.3705750415606534 act_current_ev_ch[0,0,12]
   - 0.4159374522108823 act_current_ev_ch[0,0,13]
   - 0.2968354032995391 act_current_ev_ch[0,0,14]
   - 0.3809133064208705 act_current_ev_ch[0,0,15]
   - 0.3468057619182243 act_current_ev_ch[0,0,16]
   - 0.3120123252957806 act_current_ev_ch[0,0,17]
   - 0.3173061653840341 act_current_ev_ch[0,0,18]
   - 0.3357504280236736 act_current_ev_ch[0,0,19]
   - 0.2924955767960944 act_current_ev_ch[0,0,20]
   - 0.3378517520634162 act_current_ev_ch[0,0,21]
   - 0.3295898697113126 act_current_ev_ch[0,0,22]
   - 0.40529365358821 act_current_ev_ch[0,0,23] + 0 is_exceeding_limit[0]
   + 0 is_exceeding_limit[1] + 0 is_exceeding_limit[2]
   + 0 is_exceeding_limit[3] + 0 is_exceeding_limit[4]
   + 0 is_exceeding_limit[5] + 0 is_exceeding_limit[6]
   + 0 is_exceeding_limit[7] + 0 is_exceeding_limit[8]
   + 0 is_exceeding_limit[9] + 0 is_exceeding_limit[10]
   + 0 is_exceeding_limit[11] + 0 is_exceeding_limit[12]
   + 0 is_exceeding_limit[13] + 0 is_exceeding_limit[14]
   + 0 is_exceeding_limit[15] + 0 is_exceeding_limit[16]
   + 0 is_exceeding_limit[17] + 0 is_exceeding_limit[18]
   + 0 is_exceeding_limit[19] + 0 is_exceeding_limit[20]
   + 0 is_exceeding_limit[21] + 0 is_exceeding_limit[22]
   + 0 is_exceeding_limit[23] - 100 user_satisfaction[0,0,0]
   - 100 user_satisfaction[0,0,1] - 100 user_satisfaction[0,0,2]
   - 100 user_satisfaction[0,0,3] - 100 user_satisfaction[0,0,4]
   - 100 user_satisfaction[0,0,5] - 100 user_satisfaction[0,0,6]
   - 100 user_satisfaction[0,0,7] - 100 user_satisfaction[0,0,8]
   - 100 user_satisfaction[0,0,9] - 100 user_satisfaction[0,0,10]
   - 100 user_satisfaction[0,0,11] - 100 user_satisfaction[0,0,12]
   - 100 user_satisfaction[0,0,13] - 100 user_satisfaction[0,0,14]
   - 100 user_satisfaction[0,0,15] - 100 user_satisfaction[0,0,16]
   - 100 user_satisfaction[0,0,17] - 100 user_satisfaction[0,0,18]
   - 100 user_satisfaction[0,0,19] - 100 user_satisfaction[0,0,20]
   - 100 user_satisfaction[0,0,21] - 100 user_satisfaction[0,0,22]
   - 100 user_satisfaction[0,0,23] + 0 total_soc
Subject To
 R0: - current_cs_ch[0,0] + current_tr_ch[0,0] = 0
 R1: - current_cs_dis[0,0] + current_tr_dis[0,0] = 0
 power_tr_ch.0.0: - power_cs_ch[0,0] + power_tr_ch[0,0] = 0
 power_tr_dis.0.0: - power_cs_dis[0,0] + power_tr_dis[0,0] = 0
 total_power.0: total_power[0] - power_tr_ch[0,0] + power_tr_dis[0,0] = 0
 R5: - current_cs_ch[0,1] + current_tr_ch[0,1] = 0
 R6: - current_cs_dis[0,1] + current_tr_dis[0,1] = 0
 power_tr_ch.0.1: - power_cs_ch[0,1] + power_tr_ch[0,1] = 0
 power_tr_dis.0.1: - power_cs_dis[0,1] + power_tr_dis[0,1] = 0
 total_power.1: total_power[1] - power_tr_ch[0,1] + power_tr_dis[0,1] = 0
 R10: - current_cs_ch[0,2] + current_tr_ch[0,2] = 0
 R11: - current_cs_dis[0,2] + current_tr_dis[0,2] = 0
 power_tr_ch.0.2: - power_cs_ch[0,2] + power_tr_ch[0,2] = 0
 power_tr_dis.0.2: - power_cs_dis[0,2] + power_tr_dis[0,2] = 0
 total_power.2: total_power[2] - power_tr_ch[0,2] + power_tr_dis[0,2] = 0
 R15: - current_cs_ch[0,3] + current_tr_ch[0,3] = 0
 R16: - current_cs_dis[0,3] + current_tr_dis[0,3] = 0
 power_tr_ch.0.3: - power_cs_ch[0,3] + power_tr_ch[0,3] = 0
 power_tr_dis.0.3: - power_cs_dis[0,3] + power_tr_dis[0,3] = 0
 total_power.3: total_power[3] - power_tr_ch[0,3] + power_tr_dis[0,3] = 0
 R20: - current_cs_ch[0,4] + current_tr_ch[0,4] = 0
 R21: - current_cs_dis[0,4] + current_tr_dis[0,4] = 0
 power_tr_ch.0.4: - power_cs_ch[0,4] + power_tr_ch[0,4] = 0
 power_tr_dis.0.4: - power_cs_dis[0,4] + power_tr_dis[0,4] = 0
 total_power.4: total_power[4] - power_tr_ch[0,4] + power_tr_dis[0,4] = 0
 R25: - current_cs_ch[0,5] + current_tr_ch[0,5] = 0
 R26: - current_cs_dis[0,5] + current_tr_dis[0,5] = 0
 power_tr_ch.0.5: - power_cs_ch[0,5] + power_tr_ch[0,5] = 0
 power_tr_dis.0.5: - power_cs_dis[0,5] + power_tr_dis[0,5] = 0
 total_power.5: total_power[5] - power_tr_ch[0,5] + power_tr_dis[0,5] = 0
 R30: - current_cs_ch[0,6] + current_tr_ch[0,6] = 0
 R31: - current_cs_dis[0,6] + current_tr_dis[0,6] = 0
 power_tr_ch.0.6: - power_cs_ch[0,6] + power_tr_ch[0,6] = 0
 power_tr_dis.0.6: - power_cs_dis[0,6] + power_tr_dis[0,6] = 0
 total_power.6: total_power[6] - power_tr_ch[0,6] + power_tr_dis[0,6] = 0
 R35: - current_cs_ch[0,7] + current_tr_ch[0,7] = 0
 R36: - current_cs_dis[0,7] + current_tr_dis[0,7] = 0
 power_tr_ch.0.7: - power_cs_ch[0,7] + power_tr_ch[0,7] = 0
 power_tr_dis.0.7: - power_cs_dis[0,7] + power_tr_dis[0,7] = 0
 total_power.7: total_power[7] - power_tr_ch[0,7] + power_tr_dis[0,7] = 0
 R40: - current_cs_ch[0,8] + current_tr_ch[0,8] = 0
 R41: - current_cs_dis[0,8] + current_tr_dis[0,8] = 0
 power_tr_ch.0.8: - power_cs_ch[0,8] + power_tr_ch[0,8] = 0
 power_tr_dis.0.8: - power_cs_dis[0,8] + power_tr_dis[0,8] = 0
 total_power.8: total_power[8] - power_tr_ch[0,8] + power_tr_dis[0,8] = 0
 R45: - current_cs_ch[0,9] + current_tr_ch[0,9] = 0
 R46: - current_cs_dis[0,9] + current_tr_dis[0,9] = 0
 power_tr_ch.0.9: - power_cs_ch[0,9] + power_tr_ch[0,9] = 0
 power_tr_dis.0.9: - power_cs_dis[0,9] + power_tr_dis[0,9] = 0
 total_power.9: total_power[9] - power_tr_ch[0,9] + power_tr_dis[0,9] = 0
 R50: - current_cs_ch[0,10] + current_tr_ch[0,10] = 0
 R51: - current_cs_dis[0,10] + current_tr_dis[0,10] = 0
 power_tr_ch.0.10: - power_cs_ch[0,10] + power_tr_ch[0,10] = 0
 power_tr_dis.0.10: - power_cs_dis[0,10] + power_tr_dis[0,10] = 0
 total_power.10: total_power[10] - power_tr_ch[0,10] + power_tr_dis[0,10]
   = 0
 R55: - current_cs_ch[0,11] + current_tr_ch[0,11] = 0
 R56: - current_cs_dis[0,11] + current_tr_dis[0,11] = 0
 power_tr_ch.0.11: - power_cs_ch[0,11] + power_tr_ch[0,11] = 0
 power_tr_dis.0.11: - power_cs_dis[0,11] + power_tr_dis[0,11] = 0
 total_power.11: total_power[11] - power_tr_ch[0,11] + power_tr_dis[0,11]
   = 0
 R60: - current_cs_ch[0,12] + current_tr_ch[0,12] = 0
 R61: - current_cs_dis[0,12] + current_tr_dis[0,12] = 0
 power_tr_ch.0.12: - power_cs_ch[0,12] + power_tr_ch[0,12] = 0
 power_tr_dis.0.12: - power_cs_dis[0,12] + power_tr_dis[0,12] = 0
 total_power.12: total_power[12] - power_tr_ch[0,12] + power_tr_dis[0,12]
   = 0
 R65: - current_cs_ch[0,13] + current_tr_ch[0,13] = 0
 R66: - current_cs_dis[0,13] + current_tr_dis[0,13] = 0
 power_tr_ch.0.13: - power_cs_ch[0,13] + power_tr_ch[0,13] = 0
 power_tr_dis.0.13: - power_cs_dis[0,13] + power_tr_dis[0,13] = 0
 total_power.13: total_power[13] - power_tr_ch[0,13] + power_tr_dis[0,13]
   = 0
 R70: - current_cs_ch[0,14] + current_tr_ch[0,14] = 0
 R71: - current_cs_dis[0,14] + current_tr_dis[0,14] = 0
 power_tr_ch.0.14: - power_cs_ch[0,14] + power_tr_ch[0,14] = 0
 power_tr_dis.0.14: - power_cs_dis[0,14] + power_tr_dis[0,14] = 0
 total_power.14: total_power[14] - power_tr_ch[0,14] + power_tr_dis[0,14]
   = 0
 R75: - current_cs_ch[0,15] + current_tr_ch[0,15] = 0
 R76: - current_cs_dis[0,15] + current_tr_dis[0,15] = 0
 power_tr_ch.0.15: - power_cs_ch[0,15] + power_tr_ch[0,15] = 0
 power_tr_dis.0.15: - power_cs_dis[0,15] + power_tr_dis[0,15] = 0
 total_power.15: total_power[15] - power_tr_ch[0,15] + power_tr_dis[0,15]
   = 0
 R80: - current_cs_ch[0,16] + current_tr_ch[0,16] = 0
 R81: - current_cs_dis[0,16] + current_tr_dis[0,16] = 0
 power_tr_ch.0.16: - power_cs_ch[0,16] + power_tr_ch[0,16] = 0
 power_tr_dis.0.16: - power_cs_dis[0,16] + power_tr_dis[0,16] = 0
 total_power.16: total_power[16] - power_tr_ch[0,16] + power_tr_dis[0,16]
   = 0
 R85: - current_cs_ch[0,17] + current_tr_ch[0,17] = 0
 R86: - current_cs_dis[0,17] + current_tr_dis[0,17] = 0
 power_tr_ch.0.17: - power_cs_ch[0,17] + power_tr_ch[0,17] = 0
 power_tr_dis.0.17: - power_cs_dis[0,17] + power_tr_dis[0,17] = 0
 total_power.17: total_power[17] - power_tr_ch[0,17] + power_tr_dis[0,17]
   = 0
 R90: - current_cs_ch[0,18] + current_tr_ch[0,18] = 0
 R91: - current_cs_dis[0,18] + current_tr_dis[0,18] = 0
 power_tr_ch.0.18: - power_cs_ch[0,18] + power_tr_ch[0,18] = 0
 power_tr_dis.0.18: - power_cs_dis[0,18] + power_tr_dis[0,18] = 0
 total_power.18: total_power[18] - power_tr_ch[0,18] + power_tr_dis[0,18]
   = 0
 R95: - current_cs_ch[0,19] + current_tr_ch[0,19] = 0
 R96: - current_cs_dis[0,19] + current_tr_dis[0,19] = 0
 power_tr_ch.0.19: - power_cs_ch[0,19] + power_tr_ch[0,19] = 0
 power_tr_dis.0.19: - power_cs_dis[0,19] + power_tr_dis[0,19] = 0
 total_power.19: total_power[19] - power_tr_ch[0,19] + power_tr_dis[0,19]
   = 0
 R100: - current_cs_ch[0,20] + current_tr_ch[0,20] = 0
 R101: - current_cs_dis[0,20] + current_tr_dis[0,20] = 0
 power_tr_ch.0.20: - power_cs_ch[0,20] + power_tr_ch[0,20] = 0
 power_tr_dis.0.20: - power_cs_dis[0,20] + power_tr_dis[0,20] = 0
 total_power.20: total_power[20] - power_tr_ch[0,20] + power_tr_dis[0,20]
   = 0
 R105: - current_cs_ch[0,21] + current_tr_ch[0,21] = 0
 R106: - current_cs_dis[0,21] + current_tr_dis[0,21] = 0
 power_tr_ch.0.21: - power_cs_ch[0,21] + power_tr_ch[0,21] = 0
 power_tr_dis.0.21: - power_cs_dis[0,21] + power_tr_dis[0,21] = 0
 total_power.21: total_power[21] - power_tr_ch[0,21] + power_tr_dis[0,21]
   = 0
 R110: - current_cs_ch[0,22] + current_tr_ch[0,22] = 0
 R111: - current_cs_dis[0,22] + current_tr_dis[0,22] = 0
 power_tr_ch.0.22: - power_cs_ch[0,22] + power_tr_ch[0,22] = 0
 power_tr_dis.0.22: - power_cs_dis[0,22] + power_tr_dis[0,22] = 0
 total_power.22: total_power[22] - power_tr_ch[0,22] + power_tr_dis[0,22]
   = 0
 R115: - current_cs_ch[0,23] + current_tr_ch[0,23] = 0
 R116: - current_cs_dis[0,23] + current_tr_dis[0,23] = 0
 power_tr_ch.0.23: - power_cs_ch[0,23] + power_tr_ch[0,23] = 0
 power_tr_dis.0.23: - power_cs_dis[0,23] + power_tr_dis[0,23] = 0
 total_power.23: total_power[23] - power_tr_ch[0,23] + power_tr_dis[0,23]
   = 0
 R120: - 0.6928203230275509 current_cs_ch[0,0] + power_cs_ch[0,0] = 0
 R121: - 0.6928203230275509 current_cs_ch[0,1] + power_cs_ch[0,1] = 0
 R122: - 0.6928203230275509 current_cs_ch[0,2] + power_cs_ch[0,2] = 0
 R123: - 0.6928203230275509 current_cs_ch[0,3] + power_cs_ch[0,3] = 0
 R124: - 0.6928203230275509 current_cs_ch[0,4] + power_cs_ch[0,4] = 0
 R125: - 0.6928203230275509 current_cs_ch[0,5] + power_cs_ch[0,5] = 0
 R126: - 0.6928203230275509 current_cs_ch[0,6] + power_cs_ch[0,6] = 0
 R127: - 0.6928203230275509 current_cs_ch[0,7] + power_cs_ch[0,7] = 0
 R128: - 0.6928203230275509 current_cs_ch[0,8] + power_cs_ch[0,8] = 0
 R129: - 0.6928203230275509 current_cs_ch[0,9] + power_cs_ch[0,9] = 0
 R130: - 0.6928203230275509 current_cs_ch[0,10] + power_cs_ch[0,10] = 0
 R131: - 0.6928203230275509 current_cs_ch[0,11] + power_cs_ch[0,11] = 0
 R132: - 0.6928203230275509 current_cs_ch[0,12] + power_cs_ch[0,12] = 0
 R133: - 0.6928203230275509 current_cs_ch[0,13] + power_cs_ch[0,13] = 0
 R134: - 0.6928203230275509 current_cs_ch[0,14] + power_cs_ch[0,14] = 0
 R135: - 0.6928203230275509 current_cs_ch[0,15] + power_cs_ch[0,15] = 0
 R136: - 0.6928203230275509 current_cs_ch[0,16] + power_cs_ch[0,16] = 0
 R137: - 0.6928203230275509 current_cs_ch[0,17] + power_cs_ch[0,17] = 0
 R138: - 0.6928203230275509 current_cs_ch[0,18] + power_cs_ch[0,18] = 0
 R139: - 0.6928203230275509 current_cs_ch[0,19] + power_cs_ch[0,19] = 0
 R140: - 0.6928203230275509 current_cs_ch[0,20] + power_cs_ch[0,20] = 0
 R141: - 0.6928203230275509 current_cs_ch[0,21] + power_cs_ch[0,21] = 0
 R142: - 0.6928203230275509 current_cs_ch[0,22] + power_cs_ch[0,22] = 0
 R143: - 0.6928203230275509 current_cs_ch[0,23] + power_cs_ch[0,23] = 0
 R144: - 0.6928203230275509 current_cs_dis[0,0] + power_cs_dis[0,0] = 0
 R145: - 0.6928203230275509 current_cs_dis[0,1] + power_cs_dis[0,1] = 0
 R146: - 0.6928203230275509 current_cs_dis[0,2] + power_cs_dis[0,2] = 0
 R147: - 0.6928203230275509 current_cs_dis[0,3] + power_cs_dis[0,3] = 0
 R148: - 0.6928203230275509 current_cs_dis[0,4] + power_cs_dis[0,4] = 0
 R149: - 0.6928203230275509 current_cs_dis[0,5] + power_cs_dis[0,5] = 0
 R150: - 0.6928203230275509 current_cs_dis[0,6] + power_cs_dis[0,6] = 0
 R151: - 0.6928203230275509 current_cs_dis[0,7] + power_cs_dis[0,7] = 0
 R152: - 0.6928203230275509 current_cs_dis[0,8] + power_cs_dis[0,8] = 0
 R153: - 0.6928203230275509 current_cs_dis[0,9] + power_cs_dis[0,9] = 0
 R154: - 0.6928203230275509 current_cs_dis[0,10] + power_cs_dis[0,10] = 0
 R155: - 0.6928203230275509 current_cs_dis[0,11] + power_cs_dis[0,11] = 0
 R156: - 0.6928203230275509 current_cs_dis[0,12] + power_cs_dis[0,12] = 0
 R157: - 0.6928203230275509 current_cs_dis[0,13] + power_cs_dis[0,13] = 0
 R158: - 0.6928203230275509 current_cs_dis[0,14] + power_cs_dis[0,14] = 0
 R159: - 0.6928203230275509 current_cs_dis[0,15] + power_cs_dis[0,15] = 0
 R160: - 0.6928203230275509 current_cs_dis[0,16] + power_cs_dis[0,16] = 0
 R161: - 0.6928203230275509 current_cs_dis[0,17] + power_cs_dis[0,17] = 0
 R162: - 0.6928203230275509 current_cs_dis[0,18] + power_cs_dis[0,18] = 0
 R163: - 0.6928203230275509 current_cs_dis[0,19] + power_cs_dis[0,19] = 0
 R164: - 0.6928203230275509 current_cs_dis[0,20] + power_cs_dis[0,20] = 0
 R165: - 0.6928203230275509 current_cs_dis[0,21] + power_cs_dis[0,21] = 0
 R166: - 0.6928203230275509 current_cs_dis[0,22] + power_cs_dis[0,22] = 0
 R167: - 0.6928203230275509 current_cs_dis[0,23] + power_cs_dis[0,23] = 0
 tr_current_limit_max[0,0]: current_tr_ch[0,0] - current_tr_dis[0,0]
   <= 377.3357624728399
 tr_current_limit_max[0,1]: current_tr_ch[0,1] - current_tr_dis[0,1]
   <= 385.5774426389066
 tr_current_limit_max[0,2]: current_tr_ch[0,2] - current_tr_dis[0,2]
   <= 361.2726868257917
 tr_current_limit_max[0,3]: current_tr_ch[0,3] - current_tr_dis[0,3]
   <= 382.6149481513528
 tr_current_limit_max[0,4]: current_tr_ch[0,4] - current_tr_dis[0,4]
   <= 307.8145250521119
 tr_current_limit_max[0,5]: current_tr_ch[0,5] - current_tr_dis[0,5]
   <= 326.1942918901191
 tr_current_limit_max[0,6]: current_tr_ch[0,6] - current_tr_dis[0,6]
   <= 300.5671769458815
 tr_current_limit_max[0,7]: current_tr_ch[0,7] - current_tr_dis[0,7]
   <= 263.8383958078001
 tr_current_limit_max[0,8]: current_tr_ch[0,8] - current_tr_dis[0,8]
   <= 250.327780809198
 tr_current_limit_max[0,9]: current_tr_ch[0,9] - current_tr_dis[0,9]
   <= 223.3270525039492
 tr_current_limit_max[0,10]: current_tr_ch[0,10] - current_tr_dis[0,10]
   <= 131.0694387046937
 tr_current_limit_max[0,11]: current_tr_ch[0,11] - current_tr_dis[0,11]
   <= 111.5415771171845
 tr_current_limit_max[0,12]: current_tr_ch[0,12] - current_tr_dis[0,12]
   <= 97.87719942891715
 tr_current_limit_max[0,13]: current_tr_ch[0,13] - current_tr_dis[0,13]
   <= 101.8442768222851
 tr_current_limit_max[0,14]: current_tr_ch[0,14] - current_tr_dis[0,14]
   <= 133.5193908933629
 tr_current_limit_max[0,15]: current_tr_ch[0,15] - current_tr_dis[0,15]
   <= 144.4672943975411
 tr_current_limit_max[0,16]: current_tr_ch[0,16] - current_tr_dis[0,16]
   <= 193.0768060237712
 tr_current_limit_max[0,17]: current_tr_ch[0,17] - current_tr_dis[0,17]
   <= 227.6324078145812
 tr_current_limit_max[0,18]: current_tr_ch[0,18] - current_tr_dis[0,18]
   <= 260.2403488696291
 tr_current_limit_max[0,19]: current_tr_ch[0,19] - current_tr_dis[0,19]
   <= 283.1202370918445
 tr_current_limit_max[0,20]: current_tr_ch[0,20] - current_tr_dis[0,20]
   <= 316.8352695124836
 tr_current_limit_max[0,21]: current_tr_ch[0,21] - current_tr_dis[0,21]
   <= 325.527986901569
 tr_current_limit_max[0,22]: current_tr_ch[0,22] - current_tr_dis[0,22]
   <= 359.4070328578512
 tr_current_limit_max[0,23]: current_tr_ch[0,23] - current_tr_dis[0,23]
   <= 386.9715576918732
 tr_current_limit_min[0,0]: current_tr_ch[0,0] - current_tr_dis[0,0]
   >= -377.3357624728399
 tr_current_limit_min[0,1]: current_tr_ch[0,1] - current_tr_dis[0,1]
   >= -385.5774426389066
 tr_current_limit_min[0,2]: current_tr_ch[0,2] - current_tr_dis[0,2]
   >= -361.2726868257917
 tr_current_limit_min[0,3]: current_tr_ch[0,3] - current_tr_dis[0,3]
   >= -382.6149481513528
 tr_current_limit_min[0,4]: current_tr_ch[0,4] - current_tr_dis[0,4]
   >= -307.8145250521119
 tr_current_limit_min[0,5]: current_tr_ch[0,5] - current_tr_dis[0,5]
   >= -326.1942918901191
 tr_current_limit_min[0,6]: current_tr_ch[0,6] - current_tr_dis[0,6]
   >= -300.5671769458815
 tr_current_limit_min[0,7]: current_tr_ch[0,7] - current_tr_dis[0,7]
   >= -263.8383958078001
 tr_current_limit_min[0,8]: current_tr_ch[0,8] - current_tr_dis[0,8]
   >= -250.327780809198
 tr_current_limit_min[0,9]: current_tr_ch[0,9] - current_tr_dis[0,9]
   >= -223.3270525039492
 tr_current_limit_min[0,10]: current_tr_ch[0,10] - current_tr_dis[0,10]
   >= -131.0694387046937
 tr_current_limit_min[0,11]: current_tr_ch[0,11] - current_tr_dis[0,11]
   >= -111.5415771171845
 tr_current_limit_min[0,12]: current_tr_ch[0,12] - current_tr_dis[0,12]
   >= -97.87719942891715
 tr_current_limit_min[0,13]: current_tr_ch[0,13] - current_tr_dis[0,13]
   >= -101.8442768222851
 tr_current_limit_min[0,14]: current_tr_ch[0,14] - current_tr_dis[0,14]
   >= -133.5193908933629
 tr_current_limit_min[0,15]: current_tr_ch[0,15] - current_tr_dis[0,15]
   >= -144.4672943975411
 tr_current_limit_min[0,16]: current_tr_ch[0,16] - current_tr_dis[0,16]
   >= -193.0768060237712
 tr_current_limit_min[0,17]: current_tr_ch[0,17] - current_tr_dis[0,17]
   >= -227.6324078145812
 tr_current_limit_min[0,18]: current_tr_ch[0,18] - current_tr_dis[0,18]
   >= -260.2403488696291
 tr_current_limit_min[0,19]: current_tr_ch[0,19] - current_tr_dis[0,19]
   >= -283.1202370918445
 tr_current_limit_min[0,20]: current_tr_ch[0,20] - current_tr_dis[0,20]
   >= -316.8352695124836
 tr_current_limit_min[0,21]: current_tr_ch[0,21] - current_tr_dis[0,21]
   >= -325.527986901569
 tr_current_limit_min[0,22]: current_tr_ch[0,22] - current_tr_dis[0,22]
   >= -359.4070328578512
 tr_current_limit_min[0,23]: current_tr_ch[0,23] - current_tr_dis[0,23]
   >= -386.9715576918732
 cs_ch_current_output[0,0]: - act_current_ev_ch[0,0,0] + current_cs_ch[0,0]
   = 0
 cs_ch_current_output[0,1]: - act_current_ev_ch[0,0,1] + current_cs_ch[0,1]
   = 0
 cs_ch_current_output[0,2]: - act_current_ev_ch[0,0,2] + current_cs_ch[0,2]
   = 0
 cs_ch_current_output[0,3]: - act_current_ev_ch[0,0,3] + current_cs_ch[0,3]
   = 0
 cs_ch_current_output[0,4]: - act_current_ev_ch[0,0,4] + current_cs_ch[0,4]
   = 0
 cs_ch_current_output[0,5]: - act_current_ev_ch[0,0,5] + current_cs_ch[0,5]
   = 0
 cs_ch_current_output[0,6]: - act_current_ev_ch[0,0,6] + current_cs_ch[0,6]
   = 0
 cs_ch_current_output[0,7]: - act_current_ev_ch[0,0,7] + current_cs_ch[0,7]
   = 0
 cs_ch_current_output[0,8]: - act_current_ev_ch[0,0,8] + current_cs_ch[0,8]
   = 0
 cs_ch_current_output[0,9]: - act_current_ev_ch[0,0,9] + current_cs_ch[0,9]
   = 0
 cs_ch_current_output[0,10]: - act_current_ev_ch[0,0,10]
   + current_cs_ch[0,10] = 0
 cs_ch_current_output[0,11]: - act_current_ev_ch[0,0,11]
   + current_cs_ch[0,11] = 0
 cs_ch_current_output[0,12]: - act_current_ev_ch[0,0,12]
   + current_cs_ch[0,12] = 0
 cs_ch_current_output[0,13]: - act_current_ev_ch[0,0,13]
   + current_cs_ch[0,13] = 0
 cs_ch_current_output[0,14]: - act_current_ev_ch[0,0,14]
   + current_cs_ch[0,14] = 0
 cs_ch_current_output[0,15]: - act_current_ev_ch[0,0,15]
   + current_cs_ch[0,15] = 0
 cs_ch_current_output[0,16]: - act_current_ev_ch[0,0,16]
   + current_cs_ch[0,16] = 0
 cs_ch_current_output[0,17]: - act_current_ev_ch[0,0,17]
   + current_cs_ch[0,17] = 0
 cs_ch_current_output[0,18]: - act_current_ev_ch[0,0,18]
   + current_cs_ch[0,18] = 0
 cs_ch_current_output[0,19]: - act_current_ev_ch[0,0,19]
   + current_cs_ch[0,19] = 0
 cs_ch_current_output[0,20]: - act_current_ev_ch[0,0,20]
   + current_cs_ch[0,20] = 0
 cs_ch_current_output[0,21]: - act_current_ev_ch[0,0,21]
   + current_cs_ch[0,21] = 0
 cs_ch_current_output[0,22]: - act_current_ev_ch[0,0,22]
   + current_cs_ch[0,22] = 0
 cs_ch_current_output[0,23]: - act_current_ev_ch[0,0,23]
   + current_cs_ch[0,23] = 0
 cs_dis_current_output[0,0]: - act_current_ev_dis[0,0,0]
   + current_cs_dis[0,0] = 0
 cs_dis_current_output[0,1]: - act_current_ev_dis[0,0,1]
   + current_cs_dis[0,1] = 0
 cs_dis_current_output[0,2]: - act_current_ev_dis[0,0,2]
   + current_cs_dis[0,2] = 0
 cs_dis_current_output[0,3]: - act_current_ev_dis[0,0,3]
   + current_cs_dis[0,3] = 0
 cs_dis_current_output[0,4]: - act_current_ev_dis[0,0,4]
   + current_cs_dis[0,4] = 0
 cs_dis_current_output[0,5]: - act_current_ev_dis[0,0,5]
   + current_cs_dis[0,5] = 0
 cs_dis_current_output[0,6]: - act_current_ev_dis[0,0,6]
   + current_cs_dis[0,6] = 0
 cs_dis_current_output[0,7]: - act_current_ev_dis[0,0,7]
   + current_cs_dis[0,7] = 0
 cs_dis_current_output[0,8]: - act_current_ev_dis[0,0,8]
   + current_cs_dis[0,8] = 0
 cs_dis_current_output[0,9]: - act_current_ev_dis[0,0,9]
   + current_cs_dis[0,9] = 0
 cs_dis_current_output[0,10]: - act_current_ev_dis[0,0,10]
   + current_cs_dis[0,10] = 0
 cs_dis_current_output[0,11]: - act_current_ev_dis[0,0,11]
   + current_cs_dis[0,11] = 0
 cs_dis_current_output[0,12]: - act_current_ev_dis[0,0,12]
   + current_cs_dis[0,12] = 0
 cs_dis_current_output[0,13]: - act_current_ev_dis[0,0,13]
   + current_cs_dis[0,13] = 0
 cs_dis_current_output[0,14]: - act_current_ev_dis[0,0,14]
   + current_cs_dis[0,14] = 0
 cs_dis_current_output[0,15]: - act_current_ev_dis[0,0,15]
   + current_cs_dis[0,15] = 0
 cs_dis_current_output[0,16]: - act_current_ev_dis[0,0,16]
   + current_cs_dis[0,16] = 0
 cs_dis_current_output[0,17]: - act_current_ev_dis[0,0,17]
   + current_cs_dis[0,17] = 0
 cs_dis_current_output[0,18]: - act_current_ev_dis[0,0,18]
   + current_cs_dis[0,18] = 0
 cs_dis_current_output[0,19]: - act_current_ev_dis[0,0,19]
   + current_cs_dis[0,19] = 0
 cs_dis_current_output[0,20]: - act_current_ev_dis[0,0,20]
   + current_cs_dis[0,20] = 0
 cs_dis_current_output[0,21]: - act_current_ev_dis[0,0,21]
   + current_cs_dis[0,21] = 0
 cs_dis_current_output[0,22]: - act_current_ev_dis[0,0,22]
   + current_cs_dis[0,22] = 0
 cs_dis_current_output[0,23]: - act_current_ev_dis[0,0,23]
   + current_cs_dis[0,23] = 0
 cs_current_dis_limit_max[0,0]: current_cs_ch[0,0] - current_cs_dis[0,0]
   >= -32
 cs_current_dis_limit_max[0,1]: current_cs_ch[0,1] - current_cs_dis[0,1]
   >= -32
 cs_current_dis_limit_max[0,2]: current_cs_ch[0,2] - current_cs_dis[0,2]
   >= -32
 cs_current_dis_limit_max[0,3]: current_cs_ch[0,3] - current_cs_dis[0,3]
   >= -32
 cs_current_dis_limit_max[0,4]: current_cs_ch[0,4] - current_cs_dis[0,4]
   >= -32
 cs_current_dis_limit_max[0,5]: current_cs_ch[0,5] - current_cs_dis[0,5]
   >= -32
 cs_current_dis_limit_max[0,6]: current_cs_ch[0,6] - current_cs_dis[0,6]
   >= -32
 cs_current_dis_limit_max[0,7]: current_cs_ch[0,7] - current_cs_dis[0,7]
   >= -32
 cs_current_dis_limit_max[0,8]: current_cs_ch[0,8] - current_cs_dis[0,8]
   >= -32
 cs_current_dis_limit_max[0,9]: current_cs_ch[0,9] - current_cs_dis[0,9]
   >= -32
 cs_current_dis_limit_max[0,10]: current_cs_ch[0,10] - current_cs_dis[0,10]
   >= -32
 cs_current_dis_limit_max[0,11]: current_cs_ch[0,11] - current_cs_dis[0,11]
   >= -32
 cs_current_dis_limit_max[0,12]: current_cs_ch[0,12] - current_cs_dis[0,12]
   >= -32
 cs_current_dis_limit_max[0,13]: current_cs_ch[0,13] - current_cs_dis[0,13]
   >= -32
 cs_current_dis_limit_max[0,14]: current_cs_ch[0,14] - current_cs_dis[0,14]
   >= -32
 cs_current_dis_limit_max[0,15]: current_cs_ch[0,15] - current_cs_dis[0,15]
   >= -32
 cs_current_dis_limit_max[0,16]: current_cs_ch[0,16] - current_cs_dis[0,16]
   >= -32
 cs_current_dis_limit_max[0,17]: current_cs_ch[0,17] - current_cs_dis[0,17]
   >= -32
 cs_current_dis_limit_max[0,18]: current_cs_ch[0,18] - current_cs_dis[0,18]
   >= -32
 cs_current_dis_limit_max[0,19]: current_cs_ch[0,19] - current_cs_dis[0,19]
   >= -32
 cs_current_dis_limit_max[0,20]: current_cs_ch[0,20] - current_cs_dis[0,20]
   >= -32
 cs_current_dis_limit_max[0,21]: current_cs_ch[0,21] - current_cs_dis[0,21]
   >= -32
 cs_current_dis_limit_max[0,22]: current_cs_ch[0,22] - current_cs_dis[0,22]
   >= -32
 cs_current_dis_limit_max[0,23]: current_cs_ch[0,23] - current_cs_dis[0,23]
   >= -32
 cs_curent_ch_limit_max[0,0]: current_cs_ch[0,0] - current_cs_dis[0,0]
   <= 32
 cs_curent_ch_limit_max[0,1]: current_cs_ch[0,1] - current_cs_dis[0,1]
   <= 32
 cs_curent_ch_limit_max[0,2]: current_cs_ch[0,2] - current_cs_dis[0,2]
   <= 32
 cs_curent_ch_limit_max[0,3]: current_cs_ch[0,3] - current_cs_dis[0,3]
   <= 32
 cs_curent_ch_limit_max[0,4]: current_cs_ch[0,4] - current_cs_dis[0,4]
   <= 32
 cs_curent_ch_limit_max[0,5]: current_cs_ch[0,5] - current_cs_dis[0,5]
   <= 32
 cs_curent_ch_limit_max[0,6]: current_cs_ch[0,6] - current_cs_dis[0,6]
   <= 32
 cs_curent_ch_limit_max[0,7]: current_cs_ch[0,7] - current_cs_dis[0,7]
   <= 32
 cs_curent_ch_limit_max[0,8]: current_cs_ch[0,8] - current_cs_dis[0,8]
   <= 32
 cs_curent_ch_limit_max[0,9]: current_cs_ch[0,9] - current_cs_dis[0,9]
   <= 32
 cs_curent_ch_limit_max[0,10]: current_cs_ch[0,10] - current_cs_dis[0,10]
   <= 32
 cs_curent_ch_limit_max[0,11]: current_cs_ch[0,11] - current_cs_dis[0,11]
   <= 32
 cs_curent_ch_limit_max[0,12]: current_cs_ch[0,12] - current_cs_dis[0,12]
   <= 32
 cs_curent_ch_limit_max[0,13]: current_cs_ch[0,13] - current_cs_dis[0,13]
   <= 32
 cs_curent_ch_limit_max[0,14]: current_cs_ch[0,14] - current_cs_dis[0,14]
   <= 32
 cs_curent_ch_limit_max[0,15]: current_cs_ch[0,15] - current_cs_dis[0,15]
   <= 32
 cs_curent_ch_limit_max[0,16]: current_cs_ch[0,16] - current_cs_dis[0,16]
   <= 32
 cs_curent_ch_limit_max[0,17]: current_cs_ch[0,17] - current_cs_dis[0,17]
   <= 32
 cs_curent_ch_limit_max[0,18]: current_cs_ch[0,18] - current_cs_dis[0,18]
   <= 32
 cs_curent_ch_limit_max[0,19]: current_cs_ch[0,19] - current_cs_dis[0,19]
   <= 32
 cs_curent_ch_limit_max[0,20]: current_cs_ch[0,20] - current_cs_dis[0,20]
   <= 32
 cs_curent_ch_limit_max[0,21]: current_cs_ch[0,21] - current_cs_dis[0,21]
   <= 32
 cs_curent_ch_limit_max[0,22]: current_cs_ch[0,22] - current_cs_dis[0,22]
   <= 32
 cs_curent_ch_limit_max[0,23]: current_cs_ch[0,23] - current_cs_dis[0,23]
   <= 32
 ev_current_ch_limit_min[0,0,0]: current_ev_ch[0,0,0] >= 0
 ev_current_ch_limit_min[0,0,1]: current_ev_ch[0,0,1] >= 0
 ev_current_ch_limit_min[0,0,2]: current_ev_ch[0,0,2] >= 0
 ev_current_ch_limit_min[0,0,3]: current_ev_ch[0,0,3] >= 0
 ev_current_ch_limit_min[0,0,4]: current_ev_ch[0,0,4] >= 0
 ev_current_ch_limit_min[0,0,5]: current_ev_ch[0,0,5] >= 0
 ev_current_ch_limit_min[0,0,6]: current_ev_ch[0,0,6] >= 0
 ev_current_ch_limit_min[0,0,7]: current_ev_ch[0,0,7] >= 0
 ev_current_ch_limit_min[0,0,8]: current_ev_ch[0,0,8] >= 0
 ev_current_ch_limit_min[0,0,9]: current_ev_ch[0,0,9] >= 0
 ev_current_ch_limit_min[0,0,10]: current_ev_ch[0,0,10] >= 0
 ev_current_ch_limit_min[0,0,11]: current_ev_ch[0,0,11] >= 0
 ev_current_ch_limit_min[0,0,12]: current_ev_ch[0,0,12] >= 0
 ev_current_ch_limit_min[0,0,13]: current_ev_ch[0,0,13] >= 0
 ev_current_ch_limit_min[0,0,14]: current_ev_ch[0,0,14] >= 0
 ev_current_ch_limit_min[0,0,15]: current_ev_ch[0,0,15] >= 0
 ev_current_ch_limit_min[0,0,16]: current_ev_ch[0,0,16] >= 0
 ev_current_ch_limit_min[0,0,17]: current_ev_ch[0,0,17] >= 0
 ev_current_ch_limit_min[0,0,18]: current_ev_ch[0,0,18] >= 0
 ev_current_ch_limit_min[0,0,19]: current_ev_ch[0,0,19] >= 0
 ev_current_ch_limit_min[0,0,20]: current_ev_ch[0,0,20] >= 0
 ev_current_ch_limit_min[0,0,21]: current_ev_ch[0,0,21] >= 0
 ev_current_ch_limit_min[0,0,22]: current_ev_ch[0,0,22] >= 0
 ev_current_ch_limit_min[0,0,23]: current_ev_ch[0,0,23] >= 0
 ev_current_dis_limit_min[0,0,0]: current_ev_dis[0,0,0] >= 0
 ev_current_dis_limit_min[0,0,1]: current_ev_dis[0,0,1] >= 0
 ev_current_dis_limit_min[0,0,2]: current_ev_dis[0,0,2] >= 0
 ev_current_dis_limit_min[0,0,3]: current_ev_dis[0,0,3] >= 0
 ev_current_dis_limit_min[0,0,4]: current_ev_dis[0,0,4] >= 0
 ev_current_dis_limit_min[0,0,5]: current_ev_dis[0,0,5] >= 0
 ev_current_dis_limit_min[0,0,6]: current_ev_dis[0,0,6] >= 0
 ev_current_dis_limit_min[0,0,7]: current_ev_dis[0,0,7] >= 0
 ev_current_dis_limit_min[0,0,8]: current_ev_dis[0,0,8] >= 0
 ev_current_dis_limit_min[0,0,9]: current_ev_dis[0,0,9] >= 0
 ev_current_dis_limit_min[0,0,10]: current_ev_dis[0,0,10] >= 0
 ev_current_dis_limit_min[0,0,11]: current_ev_dis[0,0,11] >= 0
 ev_current_dis_limit_min[0,0,12]: current_ev_dis[0,0,12] >= 0
 ev_current_dis_limit_min[0,0,13]: current_ev_dis[0,0,13] >= 0
 ev_current_dis_limit_min[0,0,14]: current_ev_dis[0,0,14] >= 0
 ev_current_dis_limit_min[0,0,15]: current_ev_dis[0,0,15] >= 0
 ev_current_dis_limit_min[0,0,16]: current_ev_dis[0,0,16] >= 0
 ev_current_dis_limit_min[0,0,17]: current_ev_dis[0,0,17] >= 0
 ev_current_dis_limit_min[0,0,18]: current_ev_dis[0,0,18] >= 0
 ev_current_dis_limit_min[0,0,19]: current_ev_dis[0,0,19] >= 0
 ev_current_dis_limit_min[0,0,20]: current_ev_dis[0,0,20] >= 0
 ev_current_dis_limit_min[0,0,21]: current_ev_dis[0,0,21] >= 0
 ev_current_dis_limit_min[0,0,22]: current_ev_dis[0,0,22] >= 0
 ev_current_dis_limit_min[0,0,23]: current_ev_dis[0,0,23] >= 0
 ev_current_ch_limit_max[0,0,6]: current_ev_ch[0,0,6] <= 15.87713240271471
 ev_current_ch_limit_max[0,0,7]: current_ev_ch[0,0,7] <= 15.87713240271471
 ev_current_ch_limit_max[0,0,8]: current_ev_ch[0,0,8] <= 15.87713240271471
 ev_current_ch_limit_max[0,0,9]: current_ev_ch[0,0,9] <= 15.87713240271471
 ev_current_ch_limit_max[0,0,10]: current_ev_ch[0,0,10]
   <= 15.87713240271471
 ev_current_ch_limit_max[0,0,11]: current_ev_ch[0,0,11]
   <= 15.87713240271471
 ev_current_ch_limit_max[0,0,12]: current_ev_ch[0,0,12]
   <= 15.87713240271471
 ev_current_ch_limit_max[0,0,13]: current_ev_ch[0,0,13]
   <= 15.87713240271471
 ev_current_ch_limit_max[0,0,14]: current_ev_ch[0,0,14]
   <= 15.87713240271471
 ev_current_ch_limit_max[0,0,15]: current_ev_ch[0,0,15]
   <= 15.87713240271471
 ev_current_ch_limit_max[0,0,16]: current_ev_ch[0,0,16]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,6]: current_ev_dis[0,0,6]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,7]: current_ev_dis[0,0,7]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,8]: current_ev_dis[0,0,8]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,9]: current_ev_dis[0,0,9]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,10]: current_ev_dis[0,0,10]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,11]: current_ev_dis[0,0,11]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,12]: current_ev_dis[0,0,12]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,13]: current_ev_dis[0,0,13]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,14]: current_ev_dis[0,0,14]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,15]: current_ev_dis[0,0,15]
   <= 15.87713240271471
 ev_current_dis_limit_max[0,0,16]: current_ev_dis[0,0,16]
   <= 15.87713240271471
 omega_empty_port_ch.0.0.0: omega_ch[0,0,0] = 0
 omega_empty_port_dis.0.0.0: omega_dis[0,0,0] = 0
 ev_empty_port_energy.0.0.0: energy[0,0,0] = 0
 omega_empty_port_ch.0.0.1: omega_ch[0,0,1] = 0
 omega_empty_port_dis.0.0.1: omega_dis[0,0,1] = 0
 ev_empty_port_energy.0.0.1: energy[0,0,1] = 0
 omega_empty_port_ch.0.0.2: omega_ch[0,0,2] = 0
 omega_empty_port_dis.0.0.2: omega_dis[0,0,2] = 0
 ev_empty_port_energy.0.0.2: energy[0,0,2] = 0
 omega_empty_port_ch.0.0.3: omega_ch[0,0,3] = 0
 omega_empty_port_dis.0.0.3: omega_dis[0,0,3] = 0
 ev_empty_port_energy.0.0.3: energy[0,0,3] = 0
 omega_empty_port_ch.0.0.4: omega_ch[0,0,4] = 0
 omega_empty_port_dis.0.0.4: omega_dis[0,0,4] = 0
 ev_empty_port_energy.0.0.4: energy[0,0,4] = 0
 omega_empty_port_ch.0.0.5: omega_ch[0,0,5] = 0
 omega_empty_port_dis.0.0.5: omega_dis[0,0,5] = 0
 omega_empty_port_ch.0.0.17: omega_ch[0,0,17] = 0
 omega_empty_port_dis.0.0.17: omega_dis[0,0,17] = 0
 omega_empty_port_ch.0.0.18: omega_ch[0,0,18] = 0
 omega_empty_port_dis.0.0.18: omega_dis[0,0,18] = 0
 ev_empty_port_energy.0.0.18: energy[0,0,18] = 0
 omega_empty_port_ch.0.0.19: omega_ch[0,0,19] = 0
 omega_empty_port_dis.0.0.19: omega_dis[0,0,19] = 0
 ev_empty_port_energy.0.0.19: energy[0,0,19] = 0
 omega_empty_port_ch.0.0.20: omega_ch[0,0,20] = 0
 omega_empty_port_dis.0.0.20: omega_dis[0,0,20] = 0
 ev_empty_port_energy.0.0.20: energy[0,0,20] = 0
 omega_empty_port_ch.0.0.21: omega_ch[0,0,21] = 0
 omega_empty_port_dis.0.0.21: omega_dis[0,0,21] = 0
 ev_empty_port_energy.0.0.21: energy[0,0,21] = 0
 omega_empty_port_ch.0.0.22: omega_ch[0,0,22] = 0
 omega_empty_port_dis.0.0.22: omega_dis[0,0,22] = 0
 ev_empty_port_energy.0.0.22: energy[0,0,22] = 0
 omega_empty_port_ch.0.0.23: omega_ch[0,0,23] = 0
 omega_empty_port_dis.0.0.23: omega_dis[0,0,23] = 0
 ev_empty_port_energy.0.0.23: energy[0,0,23] = 0
 ev_arrival_energy.0.0.5: energy[0,0,5] = 57
 ev_energy.0.0.6: - energy[0,0,5] + energy[0,0,6]
   + 0.6235382907247958 act_current_ev_dis[0,0,6]
   - 0.6235382907247958 act_current_ev_ch[0,0,6] = 0
 ev_energy.0.0.7: - energy[0,0,6] + energy[0,0,7]
   + 0.6235382907247958 act_current_ev_dis[0,0,7]
   - 0.6235382907247958 act_current_ev_ch[0,0,7] = 0
 ev_energy.0.0.8: - energy[0,0,7] + energy[0,0,8]
   + 0.6235382907247958 act_current_ev_dis[0,0,8]
   - 0.6235382907247958 act_current_ev_ch[0,0,8] = 0
 ev_energy.0.0.9: - energy[0,0,8] + energy[0,0,9]
   + 0.6235382907247958 act_current_ev_dis[0,0,9]
   - 0.6235382907247958 act_current_ev_ch[0,0,9] = 0
 ev_energy.0.0.10: - energy[0,0,9] + energy[0,0,10]
   + 0.6235382907247958 act_current_ev_dis[0,0,10]
   - 0.6235382907247958 act_current_ev_ch[0,0,10] = 0
 ev_energy.0.0.11: - energy[0,0,10] + energy[0,0,11]
   + 0.6235382907247958 act_current_ev_dis[0,0,11]
   - 0.6235382907247958 act_current_ev_ch[0,0,11] = 0
 ev_energy.0.0.12: - energy[0,0,11] + energy[0,0,12]
   + 0.6235382907247958 act_current_ev_dis[0,0,12]
   - 0.6235382907247958 act_current_ev_ch[0,0,12] = 0
 ev_energy.0.0.13: - energy[0,0,12] + energy[0,0,13]
   + 0.6235382907247958 act_current_ev_dis[0,0,13]
   - 0.6235382907247958 act_current_ev_ch[0,0,13] = 0
 ev_energy.0.0.14: - energy[0,0,13] + energy[0,0,14]
   + 0.6235382907247958 act_current_ev_dis[0,0,14]
   - 0.6235382907247958 act_current_ev_ch[0,0,14] = 0
 ev_energy.0.0.15: - energy[0,0,14] + energy[0,0,15]
   + 0.6235382907247958 act_current_ev_dis[0,0,15]
   - 0.6235382907247958 act_current_ev_ch[0,0,15] = 0
 ev_energy.0.0.16: - energy[0,0,15] + energy[0,0,16]
   + 0.6235382907247958 act_current_ev_dis[0,0,16]
   - 0.6235382907247958 act_current_ev_ch[0,0,16] = 0
 ev_energy.0.0.17: - energy[0,0,16] + energy[0,0,17]
   + 0.6235382907247958 act_current_ev_dis[0,0,17]
   - 0.6235382907247958 act_current_ev_ch[0,0,17] = 0
 ev_energy_level_min[0,0,0]: energy[0,0,0] >= 0
 ev_energy_level_min[0,0,1]: energy[0,0,1] >= 0
 ev_energy_level_min[0,0,2]: energy[0,0,2] >= 0
 ev_energy_level_min[0,0,3]: energy[0,0,3] >= 0
 ev_energy_level_min[0,0,4]: energy[0,0,4] >= 0
 ev_energy_level_min[0,0,5]: energy[0,0,5] >= 0
 ev_energy_level_min[0,0,6]: energy[0,0,6] >= 0
 ev_energy_level_min[0,0,7]: energy[0,0,7] >= 0
 ev_energy_level_min[0,0,8]: energy[0,0,8] >= 0
 ev_energy_level_min[0,0,9]: energy[0,0,9] >= 0
 ev_energy_level_min[0,0,10]: energy[0,0,10] >= 0
 ev_energy_level_min[0,0,11]: energy[0,0,11] >= 0
 ev_energy_level_min[0,0,12]: energy[0,0,12] >= 0
 ev_energy_level_min[0,0,13]: energy[0,0,13] >= 0
 ev_energy_level_min[0,0,14]: energy[0,0,14] >= 0
 ev_energy_level_min[0,0,15]: energy[0,0,15] >= 0
 ev_energy_level_min[0,0,16]: energy[0,0,16] >= 0
 ev_energy_level_min[0,0,17]: energy[0,0,17] >= 0
 ev_energy_level_min[0,0,18]: energy[0,0,18] >= 0
 ev_energy_level_min[0,0,19]: energy[0,0,19] >= 0
 ev_energy_level_min[0,0,20]: energy[0,0,20] >= 0
 ev_energy_level_min[0,0,21]: energy[0,0,21] >= 0
 ev_energy_level_min[0,0,22]: energy[0,0,22] >= 0
 ev_energy_level_min[0,0,23]: energy[0,0,23] >= 0
 ev_energy_level_max[0,0,0]: energy[0,0,0] <= 0
 ev_energy_level_max[0,0,1]: energy[0,0,1] <= 0
 ev_energy_level_max[0,0,2]: energy[0,0,2] <= 0
 ev_energy_level_max[0,0,3]: energy[0,0,3] <= 0
 ev_energy_level_max[0,0,4]: energy[0,0,4] <= 0
 ev_energy_level_max[0,0,5]: energy[0,0,5] <= 64.8
 ev_energy_level_max[0,0,6]: energy[0,0,6] <= 64.8
 ev_energy_level_max[0,0,7]: energy[0,0,7] <= 64.8
 ev_energy_level_max[0,0,8]: energy[0,0,8] <= 64.8
 ev_energy_level_max[0,0,9]: energy[0,0,9] <= 64.8
 ev_energy_level_max[0,0,10]: energy[0,0,10] <= 64.8
 ev_energy_level_max[0,0,11]: energy[0,0,11] <= 64.8
 ev_energy_level_max[0,0,12]: energy[0,0,12] <= 64.8
 ev_energy_level_max[0,0,13]: energy[0,0,13] <= 64.8
 ev_energy_level_max[0,0,14]: energy[0,0,14] <= 64.8
 ev_energy_level_max[0,0,15]: energy[0,0,15] <= 64.8
 ev_energy_level_max[0,0,16]: energy[0,0,16] <= 64.8
 ev_energy_level_max[0,0,18]: energy[0,0,18] <= 0
 ev_energy_level_max[0,0,19]: energy[0,0,19] <= 0
 ev_energy_level_max[0,0,20]: energy[0,0,20] <= 0
 ev_energy_level_max[0,0,21]: energy[0,0,21] <= 0
 ev_energy_level_max[0,0,22]: energy[0,0,22] <= 0
 ev_energy_level_max[0,0,23]: energy[0,0,23] <= 0
 act_ev_current_ch[0,0,0]: act_current_ev_ch[0,0,0] + [
   - current_ev_ch[0,0,0] * omega_ch[0,0,0] ] = 0
 act_ev_current_ch[0,0,1]: act_current_ev_ch[0,0,1] + [
   - current_ev_ch[0,0,1] * omega_ch[0,0,1] ] = 0
 act_ev_current_ch[0,0,2]: act_current_ev_ch[0,0,2] + [
   - current_ev_ch[0,0,2] * omega_ch[0,0,2] ] = 0
 act_ev_current_ch[0,0,3]: act_current_ev_ch[0,0,3] + [
   - current_ev_ch[0,0,3] * omega_ch[0,0,3] ] = 0
 act_ev_current_ch[0,0,4]: act_current_ev_ch[0,0,4] + [
   - current_ev_ch[0,0,4] * omega_ch[0,0,4] ] = 0
 act_ev_current_ch[0,0,5]: act_current_ev_ch[0,0,5] + [
   - current_ev_ch[0,0,5] * omega_ch[0,0,5] ] = 0
 act_ev_current_ch[0,0,6]: act_current_ev_ch[0,0,6] + [
   - current_ev_ch[0,0,6] * omega_ch[0,0,6] ] = 0
 act_ev_current_ch[0,0,7]: act_current_ev_ch[0,0,7] + [
   - current_ev_ch[0,0,7] * omega_ch[0,0,7] ] = 0
 act_ev_current_ch[0,0,8]: act_current_ev_ch[0,0,8] + [
   - current_ev_ch[0,0,8] * omega_ch[0,0,8] ] = 0
 act_ev_current_ch[0,0,9]: act_current_ev_ch[0,0,9] + [
   - current_ev_ch[0,0,9] * omega_ch[0,0,9] ] = 0
 act_ev_current_ch[0,0,10]: act_current_ev_ch[0,0,10] + [
   - current_ev_ch[0,0,10] * omega_ch[0,0,10] ] = 0
 act_ev_current_ch[0,0,11]: act_current_ev_ch[0,0,11] + [
   - current_ev_ch[0,0,11] * omega_ch[0,0,11] ] = 0
 act_ev_current_ch[0,0,12]: act_current_ev_ch[0,0,12] + [
   - current_ev_ch[0,0,12] * omega_ch[0,0,12] ] = 0
 act_ev_current_ch[0,0,13]: act_current_ev_ch[0,0,13] + [
   - current_ev_ch[0,0,13] * omega_ch[0,0,13] ] = 0
 act_ev_current_ch[0,0,14]: act_current_ev_ch[0,0,14] + [
   - current_ev_ch[0,0,14] * omega_ch[0,0,14] ] = 0
 act_ev_current_ch[0,0,15]: act_current_ev_ch[0,0,15] + [
   - current_ev_ch[0,0,15] * omega_ch[0,0,15] ] = 0
 act_ev_current_ch[0,0,16]: act_current_ev_ch[0,0,16] + [
   - current_ev_ch[0,0,16] * omega_ch[0,0,16] ] = 0
 act_ev_current_ch[0,0,17]: act_current_ev_ch[0,0,17] + [
   - current_ev_ch[0,0,17] * omega_ch[0,0,17] ] = 0
 act_ev_current_ch[0,0,18]: act_current_ev_ch[0,0,18] + [
   - current_ev_ch[0,0,18] * omega_ch[0,0,18] ] = 0
 act_ev_current_ch[0,0,19]: act_current_ev_ch[0,0,19] + [
   - current_ev_ch[0,0,19] * omega_ch[0,0,19] ] = 0
 act_ev_current_ch[0,0,20]: act_current_ev_ch[0,0,20] + [
   - current_ev_ch[0,0,20] * omega_ch[0,0,20] ] = 0
 act_ev_current_ch[0,0,21]: act_current_ev_ch[0,0,21] + [
   - current_ev_ch[0,0,21] * omega_ch[0,0,21] ] = 0
 act_ev_current_ch[0,0,22]: act_current_ev_ch[0,0,22] + [
   - current_ev_ch[0,0,22] * omega_ch[0,0,22] ] = 0
 act_ev_current_ch[0,0,23]: act_current_ev_ch[0,0,23] + [
   - current_ev_ch[0,0,23] * omega_ch[0,0,23] ] = 0
 act_ev_current_dis[0,0,0]: act_current_ev_dis[0,0,0] + [
   - current_ev_dis[0,0,0] * omega_dis[0,0,0] ] = 0
 act_ev_current_dis[0,0,1]: act_current_ev_dis[0,0,1] + [
   - current_ev_dis[0,0,1] * omega_dis[0,0,1] ] = 0
 act_ev_current_dis[0,0,2]: act_current_ev_dis[0,0,2] + [
   - current_ev_dis[0,0,2] * omega_dis[0,0,2] ] = 0
 act_ev_current_dis[0,0,3]: act_current_ev_dis[0,0,3] + [
   - current_ev_dis[0,0,3] * omega_dis[0,0,3] ] = 0
 act_ev_current_dis[0,0,4]: act_current_ev_dis[0,0,4] + [
   - current_ev_dis[0,0,4] * omega_dis[0,0,4] ] = 0
 act_ev_current_dis[0,0,5]: act_current_ev_dis[0,0,5] + [
   - current_ev_dis[0,0,5] * omega_dis[0,0,5] ] = 0
 act_ev_current_dis[0,0,6]: act_current_ev_dis[0,0,6] + [
   - current_ev_dis[0,0,6] * omega_dis[0,0,6] ] = 0
 act_ev_current_dis[0,0,7]: act_current_ev_dis[0,0,7] + [
   - current_ev_dis[0,0,7] * omega_dis[0,0,7] ] = 0
 act_ev_current_dis[0,0,8]: act_current_ev_dis[0,0,8] + [
   - current_ev_dis[0,0,8] * omega_dis[0,0,8] ] = 0
 act_ev_current_dis[0,0,9]: act_current_ev_dis[0,0,9] + [
   - current_ev_dis[0,0,9] * omega_dis[0,0,9] ] = 0
 act_ev_current_dis[0,0,10]: act_current_ev_dis[0,0,10] + [
   - current_ev_dis[0,0,10] * omega_dis[0,0,10] ] = 0
 act_ev_current_dis[0,0,11]: act_current_ev_dis[0,0,11] + [
   - current_ev_dis[0,0,11] * omega_dis[0,0,11] ] = 0
 act_ev_current_dis[0,0,12]: act_current_ev_dis[0,0,12] + [
   - current_ev_dis[0,0,12] * omega_dis[0,0,12] ] = 0
 act_ev_current_dis[0,0,13]: act_current_ev_dis[0,0,13] + [
   - current_ev_dis[0,0,13] * omega_dis[0,0,13] ] = 0
 act_ev_current_dis[0,0,14]: act_current_ev_dis[0,0,14] + [
   - current_ev_dis[0,0,14] * omega_dis[0,0,14] ] = 0
 act_ev_current_dis[0,0,15]: act_current_ev_dis[0,0,15] + [
   - current_ev_dis[0,0,15] * omega_dis[0,0,15] ] = 0
 act_ev_current_dis[0,0,16]: act_current_ev_dis[0,0,16] + [
   - current_ev_dis[0,0,16] * omega_dis[0,0,16] ] = 0
 act_ev_current_dis[0,0,17]: act_current_ev_dis[0,0,17] + [
   - current_ev_dis[0,0,17] * omega_dis[0,0,17] ] = 0
 act_ev_current_dis[0,0,18]: act_current_ev_dis[0,0,18] + [
   - current_ev_dis[0,0,18] * omega_dis[0,0,18] ] = 0
 act_ev_current_dis[0,0,19]: act_current_ev_dis[0,0,19] + [
   - current_ev_dis[0,0,19] * omega_dis[0,0,19] ] = 0
 act_ev_current_dis[0,0,20]: act_current_ev_dis[0,0,20] + [
   - current_ev_dis[0,0,20] * omega_dis[0,0,20] ] = 0
 act_ev_current_dis[0,0,21]: act_current_ev_dis[0,0,21] + [
   - current_ev_dis[0,0,21] * omega_dis[0,0,21] ] = 0
 act_ev_current_dis[0,0,22]: act_current_ev_dis[0,0,22] + [
   - current_ev_dis[0,0,22] * omega_dis[0,0,22] ] = 0
 act_ev_current_dis[0,0,23]: act_current_ev_dis[0,0,23] + [
   - current_ev_dis[0,0,23] * omega_dis[0,0,23] ] = 0
 ev_power_mode_2[0,0,0]: [ omega_ch[0,0,0] * omega_dis[0,0,0] ] = 0
 ev_power_mode_2[0,0,1]: [ omega_ch[0,0,1] * omega_dis[0,0,1] ] = 0
 ev_power_mode_2[0,0,2]: [ omega_ch[0,0,2] * omega_dis[0,0,2] ] = 0
 ev_power_mode_2[0,0,3]: [ omega_ch[0,0,3] * omega_dis[0,0,3] ] = 0
 ev_power_mode_2[0,0,4]: [ omega_ch[0,0,4] * omega_dis[0,0,4] ] = 0
 ev_power_mode_2[0,0,5]: [ omega_ch[0,0,5] * omega_dis[0,0,5] ] = 0
 ev_power_mode_2[0,0,6]: [ omega_ch[0,0,6] * omega_dis[0,0,6] ] = 0
 ev_power_mode_2[0,0,7]: [ omega_ch[0,0,7] * omega_dis[0,0,7] ] = 0
 ev_power_mode_2[0,0,8]: [ omega_ch[0,0,8] * omega_dis[0,0,8] ] = 0
 ev_power_mode_2[0,0,9]: [ omega_ch[0,0,9] * omega_dis[0,0,9] ] = 0
 ev_power_mode_2[0,0,10]: [ omega_ch[0,0,10] * omega_dis[0,0,10] ] = 0
 ev_power_mode_2[0,0,11]: [ omega_ch[0,0,11] * omega_dis[0,0,11] ] = 0
 ev_power_mode_2[0,0,12]: [ omega_ch[0,0,12] * omega_dis[0,0,12] ] = 0
 ev_power_mode_2[0,0,13]: [ omega_ch[0,0,13] * omega_dis[0,0,13] ] = 0
 ev_power_mode_2[0,0,14]: [ omega_ch[0,0,14] * omega_dis[0,0,14] ] = 0
 ev_power_mode_2[0,0,15]: [ omega_ch[0,0,15] * omega_dis[0,0,15] ] = 0
 ev_power_mode_2[0,0,16]: [ omega_ch[0,0,16] * omega_dis[0,0,16] ] = 0
 ev_power_mode_2[0,0,17]: [ omega_ch[0,0,17] * omega_dis[0,0,17] ] = 0
 ev_power_mode_2[0,0,18]: [ omega_ch[0,0,18] * omega_dis[0,0,18] ] = 0
 ev_power_mode_2[0,0,19]: [ omega_ch[0,0,19] * omega_dis[0,0,19] ] = 0
 ev_power_mode_2[0,0,20]: [ omega_ch[0,0,20] * omega_dis[0,0,20] ] = 0
 ev_power_mode_2[0,0,21]: [ omega_ch[0,0,21] * omega_dis[0,0,21] ] = 0
 ev_power_mode_2[0,0,22]: [ omega_ch[0,0,22] * omega_dis[0,0,22] ] = 0
 ev_power_mode_2[0,0,23]: [ omega_ch[0,0,23] * omega_dis[0,0,23] ] = 0
 ev_user_satisfaction.0.0.17: 8398.08 energy[0,0,17]
   + user_satisfaction[0,0,17] + [ - energy[0,0,17] ^2 ]
   = 1.7631936921599999e+07
Bounds
Binaries
 omega_ch[0,0,0] omega_ch[0,0,1] omega_ch[0,0,2] omega_ch[0,0,3]
 omega_ch[0,0,4] omega_ch[0,0,5] omega_ch[0,0,6] omega_ch[0,0,7]
 omega_ch[0,0,8] omega_ch[0,0,9] omega_ch[0,0,10] omega_ch[0,0,11]
 omega_ch[0,0,12] omega_ch[0,0,13] omega_ch[0,0,14] omega_ch[0,0,15]
 omega_ch[0,0,16] omega_ch[0,0,17] omega_ch[0,0,18] omega_ch[0,0,19]
 omega_ch[0,0,20] omega_ch[0,0,21] omega_ch[0,0,22] omega_ch[0,0,23]
 omega_dis[0,0,0] omega_dis[0,0,1] omega_dis[0,0,2] omega_dis[0,0,3]
 omega_dis[0,0,4] omega_dis[0,0,5] omega_dis[0,0,6] omega_dis[0,0,7]
 omega_dis[0,0,8] omega_dis[0,0,9] omega_dis[0,0,10] omega_dis[0,0,11]
 omega_dis[0,0,12] omega_dis[0,0,13] omega_dis[0,0,14] omega_dis[0,0,15]
 omega_dis[0,0,16] omega_dis[0,0,17] omega_dis[0,0,18] omega_dis[0,0,19]
 omega_dis[0,0,20] omega_dis[0,0,21] omega_dis[0,0,22] omega_dis[0,0,23]
 is_exceeding_limit[0] is_exceeding_limit[1] is_exceeding_limit[2]
 is_exceeding_limit[3] is_exceeding_limit[4] is_exceeding_limit[5]
 is_exceeding_limit[6] is_exceeding_limit[7] is_exceeding_limit[8]
 is_exceeding_limit[9] is_exceeding_limit[10] is_exceeding_limit[11]
 is_exceeding_limit[12] is_exceeding_limit[13] is_exceeding_limit[14]
 is_exceeding_limit[15] is_exceeding_limit[16] is_exceeding_limit[17]
 is_exceeding_limit[18] is_exceeding_limit[19] is_exceeding_limit[20]
 is_exceeding_limit[21] is_exceeding_limit[22] is_exceeding_limit[23]
End
