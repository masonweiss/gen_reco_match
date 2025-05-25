import ROOT
import array
import numpy as np

ROOT.gErrorIgnoreLevel = ROOT.kFatal

print("===========\nBEGIN GEN-RECO MATCHING\nLAUNCHING ROOT WITH VERSION : " + ROOT.__version__ + "\n===========")
fnames = ["imcc_1", "imcc_10", "imcc_100",
          "imcc_1k", "imcc_10k", "imcc_100k"]
dR_threshold = 1e-1  # acceptable for a match
total_unmatched_muons = []
hadron_sum_e_info = []
hadron_sum_e_minus_pz_info = []
h = 12 # use 12 decimals in csv file
verbose = False  # print out info 
print_only_smallest_displaced_muons = True # THIS FLAG WAS ADDED 5/13 <- if True, this only prints the muon in the event with the smallest displacement

for i in range(len(fnames)):
    f = fnames[i]
    file = ROOT.TFile.Open("in/"+f+".root")
    tree = file.Get("evt")
    sample_gen_muons = 0
    sample_rec_muons = 0
    unmatched_muons = []

    num_events = tree.GetEntries()
    print(f"Processing file: in/{f}.root with {num_events} entries")
    #num_events = 10

    results = np.zeros((num_events, 7))     # dR x y z eta phi pt

    for j in range(num_events):
        tree.GetEntry(j)
        dR_table = []
        
        ##################
        # FIND GEN MUONS #
        ##################
        num_gen_particles = len(tree.gen_eta)
        g_pdgids = list(tree.gen_pdgid)
        g_etas = list(tree.gen_eta)
        g_phis = list(tree.gen_phi)
        gen_muons = []  # list of lists - stores (idx, eta, phi, ...) of each gen muon
        for k in range(num_gen_particles):
            if list(tree.gen_pdgid)[k] == 13:
                g_pdgid = g_pdgids[k]
                g_eta = g_etas[k]
                g_phi = g_phis[k]
                gen_muons.append([k, list(tree.gen_eta)[k], list(tree.gen_phi)[k], list(tree.gen_pt)[k], list(tree.gen_vx)[k], list(tree.gen_vy)[k], list(tree.gen_vz)[k]])
        num_gen_muons = len(gen_muons)
        sample_gen_muons += num_gen_muons
        
        ##################
        # FIND REC MUONS #
        ##################
        num_reco_particles = len(tree.rec_eta)
        r_charges = list(tree.rec_charge)
        r_masses = list(tree.rec_mass)
        r_etas = list(tree.rec_eta)
        r_phis = list(tree.rec_phi)
        # compare each reco muon to each gen muon
        for k in range(num_reco_particles):
            # see reco muons
            if np.abs(r_masses[k] - 0.105) < 0.005: # and r_charges[k] < 0: -accept muon and antimuon
                # muon identified
                sample_rec_muons += 1
                r_eta = list(tree.rec_eta)[k]
                r_phi = list(tree.rec_phi)[k]
                for m in range(num_gen_muons):
                    g_mu_idx = gen_muons[m][0]
                    g_eta = gen_muons[m][1]
                    g_phi = gen_muons[m][2]
                    dR = np.sqrt((g_eta - r_eta)**2 + (g_phi - r_phi)**2)
                    dR_table.append([dR, j, g_mu_idx, k, g_eta, r_eta, gen_muons[m][4]])

        ##################
        #   MATCH MUONS  #
        ##################
        if len(dR_table) > 0:
            dR_table = np.array(dR_table)
            dR_table_sorted_idxs = np.argsort(dR_table[:, 0])

            num_matched = 0
            matched_muons = np.zeros((num_gen_muons, 2))  # g_mu_idx, rec_mu_idx

            k = 0
            while num_matched < num_gen_muons and k < len(dR_table_sorted_idxs):
                info = dR_table[dR_table_sorted_idxs[k]]
                dR = info[0]
                if info[2] in matched_muons[:, 0] or info[3] in matched_muons[:, 1]:
                    # PREVIOUSLY SEEN
                    pass
                elif dR < dR_threshold:
                    # GOOD MATCH
                    matched_muons[num_matched] = [info[2], info[3]]
                    num_matched += 1 # remove this item from the possible dr table AND anything with same reco_muon
                else:
                    # NOT A MATCH - DOESN'T MEET THRESHOLD
                    #print(info)  # this is unmatched !
                    num_matched += 1
                k += 1

        ####################
        # RECORD UNMATCHED #
        ####################
        min_displ = 1e3
        curr_displ = 1e3
        min_displ_idx = 0
        for k in range(len(gen_muons)):
            if len(dR_table) == 0 or gen_muons[k][0] not in matched_muons[:, 0]:

                if print_only_smallest_displaced_muons: # find the gen muon that went unmatched that had smallest displacement
                    curr_displ = gen_muons[k][4]**2 + gen_muons[k][5]**2 + gen_muons[k][6]**2
                    
                    if curr_displ < min_displ:
                        min_displ = curr_displ
                        min_displ_idx = k
                
                else: # store all unmatched
                    unmatched_muons.append([i, j]+gen_muons[k]+[tree.Q2, tree.x, tree.y])

                if verbose and i > 4 and np.abs(gen_muons[k][1]) < 1.4:
                    print(f"======== EVENT {j} =======")
                    print(dR_table)
                    print(gen_muons)
        
        if print_only_smallest_displaced_muons and curr_displ != 1e3: # store only the muon corresponding to min displacement
            unmatched_muons.append([i, j]+gen_muons[min_displ_idx]+[tree.Q2, tree.x, tree.y])

    print(f"    {len(unmatched_muons)} unmatched muons out of {sample_gen_muons} gen muons and {sample_rec_muons} rec muons")
    total_unmatched_muons.append(unmatched_muons)

    #########################
    #  RECORD HADRON TOTALS #
    #########################
    sum_e_list = []
    sum_e_minus_pz_list = []
    unmatched_muons_arr = np.array(unmatched_muons)
    events_with_unmatched = unmatched_muons_arr[:,1]

    for j_fl in events_with_unmatched:
        j = int(j_fl)
        tree.GetEntry(j)

        num_reco_particles = len(tree.rec_eta)
        r_eta = np.array(list(tree.rec_eta))
        r_mass = np.array(list(tree.rec_mass))
        r_phi = np.array(list(tree.rec_phi))
        r_pt = np.array(list(tree.rec_pt))

        r_px = r_pt * np.cos(r_phi)
        r_py = r_pt * np.sin(r_phi)
        r_pz = r_pt * np.sinh(r_eta)
        r_e = np.sqrt(np.square(r_px)+np.square(r_py)+np.square(r_pz)+np.square(r_mass))

        sum_e = np.sum(r_e)
        sum_px = np.sum(r_px)
        sum_py = np.sum(r_py)
        sum_pz = np.sum(r_pz)
        sum_e_minus_pz = sum_e - sum_pz

        sum_e_list.append(sum_e)
        sum_e_minus_pz_list.append(sum_e_minus_pz)

    hadron_sum_e_info.append(sum_e_list)
    hadron_sum_e_minus_pz_info.append(sum_e_minus_pz_list)

    # JB needs
    # sum_e_minus_pz
    # sum_pt_squared
    
    # DA needs
    # sum_pt_squared
    # sum_e_minus_pz_ (squared)

print("===========\nMATCHING RESULTS\n===========")

###################
# PRINT UNMATCHED #
###################

with open(f"out/all_unmatched.csv", "w") as csvfile_full:
    csvfile_full.write("file_idx,event_idx,gen_particle_idx,eta,phi,pt,vx,vy,vz,sum_e,sum_e_minus_pt,Q2,x,y\n")
    file_all = ROOT.TFile(f"out/all_unmatched.root", "RECREATE")
    tree_all = ROOT.TTree("unmatched", "tree storing file_idx, event_idx, eta, phi, pt, vx, vy, vz, sum_e, sum_e_minus_pt, Q2, x, y")

    f_idx_full = array.array('f', [0.])
    e_idx_full = array.array('f', [0.])
    p_idx_full = array.array('f', [0.])
    e_full = array.array('f', [0.])
    p_full = array.array('f', [0.])
    t_full = array.array('f', [0.])
    x_full = array.array('f', [0.])
    y_full = array.array('f', [0.])
    z_full = array.array('f', [0.])
    had_sum_e_full = array.array('f', [0.])
    had_sum_e_minus_pz_full = array.array('f', [0.])
    DIS_Q2_full = array.array('f', [0.])
    DIS_X_full = array.array('f', [0.])
    DIS_Y_full = array.array('f', [0.])

    tree_all.Branch("file_idx", f_idx_full, "file_idx/F")
    tree_all.Branch("event_idx", e_idx_full, "event_idx/F")
    tree_all.Branch("particle_idx", p_idx_full, "particle_idx/F")
    tree_all.Branch("eta", e_full, "eta/F")
    tree_all.Branch("phi", p_full, "phi/F")
    tree_all.Branch("pt",  t_full,  "pt/F")
    tree_all.Branch("vx",  x_full,  "vx/F")
    tree_all.Branch("vy",  y_full,  "vy/F")
    tree_all.Branch("vz",  z_full,  "vz/F")
    tree_all.Branch("sum_e",  had_sum_e_full,  "sum_e/F")
    tree_all.Branch("sum_e_minus_pt",  had_sum_e_minus_pz_full,  "sum_e_minus_pt/F")
    tree_all.Branch("Q2",  DIS_Q2_full,  "Q2/F")
    tree_all.Branch("x",  DIS_X_full,  "x/F")
    tree_all.Branch("y",  DIS_Y_full,  "y/F")

    for i in range(len(fnames)):
        f = fnames[i]
        this_file_unmatched = total_unmatched_muons[i]
        print(f"Processing file: out/{f}_unmatched.root with {len(this_file_unmatched)} entries")

        with open(f"out/{f}_unmatched.csv", "w") as csvfile:
            csvfile.write("file_idx,event_idx,gen_particle_idx,eta,phi,pt,vx,vy,vz,sum_e,sum_e_minus_pt,Q2,x,y\n")
            for p in range(len(this_file_unmatched)):
                g = this_file_unmatched[p]
                csvfile.write(f"{i},{g[1]},{g[2]},{round(g[3], h)},{round(g[4], h)},{round(g[5], h)},{round(g[6], h)},{round(g[7], h)},{round(g[8], h)},{round(hadron_sum_e_info[i][p], h)},{round(hadron_sum_e_minus_pz_info[i][p], h)},{round(g[9], h)},{round(g[10], h)},{round(g[11], h)}\n")
                csvfile_full.write(f"{i},{g[1]},{g[2]},{round(g[3], h)},{round(g[4], h)},{round(g[5], h)},{round(g[6], h)},{round(g[7], h)},{round(g[8], h)},{round(hadron_sum_e_info[i][p], h)},{round(hadron_sum_e_minus_pz_info[i][p], h)},{round(g[9], h)},{round(g[10], h)},{round(g[11], h)}\n")
            
        file = ROOT.TFile(f"out/{f}_unmatched.root", "RECREATE")
        
        # make new tree
        tree = ROOT.TTree("unmatched", "tree storing file_idx, event_idx, eta, phi, pt, vx, vy, vz, sum_e, sum_e_minus_pt, Q2, x, y")

        # single-element arrays, will store info for branch
        file_idx = array.array('f', [0.])
        event_idx = array.array('f', [0.])
        particle_idx = array.array('f', [0.])
        eta = array.array('f', [0.])
        phi = array.array('f', [0.])
        pt  = array.array('f', [0.])
        vx  = array.array('f', [0.])
        vy  = array.array('f', [0.])
        vz  = array.array('f', [0.])
        had_sum_e = array.array('f', [0.])
        had_sum_e_minus_pz = array.array('f', [0.])
        DIS_Q2 = array.array('f', [0.])
        DIS_X = array.array('f', [0.])
        DIS_Y = array.array('f', [0.])

        # new branches
        tree.Branch("file_idx", file_idx, "file_idx/F")
        tree.Branch("event_idx", event_idx, "event_idx/F")
        tree.Branch("particle_idx", particle_idx, "particle_idx/F")
        tree.Branch("eta", eta, "eta/F")
        tree.Branch("phi", phi, "phi/F")
        tree.Branch("pt",  pt,  "pt/F")
        tree.Branch("vx",  vx,  "vx/F")
        tree.Branch("vy",  vy,  "vy/F")
        tree.Branch("vz",  vz,  "vz/F")
        tree.Branch("sum_e",  had_sum_e,  "sum_e/F")
        tree.Branch("sum_e_minus_pt",  had_sum_e_minus_pz,  "sum_e_minus_pt/F")
        tree.Branch("Q2",  DIS_Q2,  "Q2/F")
        tree.Branch("x",  DIS_X,  "x/F")
        tree.Branch("y",  DIS_Y,  "y/F")

        # fill tree by looping over all entries
        count = 0
        for entry in this_file_unmatched:
            file_idx[0], event_idx[0], particle_idx[0], eta[0], phi[0], pt[0], vx[0], vy[0], vz[0], DIS_Q2[0], DIS_X[0], DIS_Y[0] = entry
            f_idx_full[0], e_idx_full[0], p_idx_full[0], e_full[0], p_full[0], t_full[0], x_full[0], y_full[0], z_full[0], DIS_Q2_full[0], DIS_X_full[0], DIS_Y_full[0] = entry

            had_sum_e_full[0] = hadron_sum_e_info[i][count]
            had_sum_e[0] = hadron_sum_e_info[i][count]

            had_sum_e_minus_pz_full[0] = hadron_sum_e_minus_pz_info[i][count]
            had_sum_e_minus_pz[0] = hadron_sum_e_minus_pz_info[i][count]
            
            tree.Fill()
            tree_all.Fill()

            count += 1

        tree.Write()
        file.Close()
    
    file_all.cd()
    tree_all.Write()
    file_all.Close()
                    