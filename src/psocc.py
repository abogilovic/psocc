import os
import time
import math
from selenium import webdriver
import pandas as pd
import matplotlib.pyplot as plt

project_path = "/home/gillabo/Desktop/psocc"
download_path = "/home/gillabo/Downloads"
leagues_id = {"NJEMAČKA 1.LIGA": 0}
#ODOKATIVNI KOEFICIJENTI; NO ML
k = [160, 100, 2.5, 5, 1, 2.5, 5, 10, 1.1, 0.9, 1.1, 0.9, 0.5, 2, 1.25, 1, 160, 0.65, 0.175, 0.175, 0.5]


def update_league_data(league_id, n_last_seasons):
    file_strt = league_id * 5
    with open(project_path + "/res/leagues.metadata", "r") as file:
        info = file.readlines()

    driver.get(info[file_strt + 0])
    upd = driver.find_element_by_tag_name("i")
    ret = {}

    league_path = "{}/res/{}".format(project_path, info[file_strt + 3].rstrip())

    if upd.text != info[file_strt + 4].rstrip() or not os.path.exists(league_path) or \
            len(os.listdir("{}/res/{}".format(project_path, info[file_strt+3].rstrip())))!=n_last_seasons:
        league_seasons = driver.find_elements_by_link_text(info[file_strt + 1].rstrip())

        for i in range(n_last_seasons):
            league_seasons[i].click()
            time.sleep(1)
            old_name = info[file_strt + 2].rstrip()
            splt = league_seasons[i].get_attribute("href").split("/")
            new_name = "{}-season-{}".format(splt[-2], splt[-1])
            os.renames("{}/{}".format(download_path, old_name), "{}/{}".format(download_path, new_name))
            if not os.path.exists(league_path):
                os.makedirs(league_path)
            os.replace("{}/{}".format(download_path, new_name), "{}/{}".format(league_path, new_name))

        info[file_strt + 4] = upd.text + "\n"
        with open(project_path + "/res/leagues.metadata", "w") as file:
            file.writelines(info)
        ret["updated"] = True
    else:
        ret["updated"] = False

    ret["text"] = upd.text
    return ret

def premier_to_data_team_name(name, league_id):
    file_strt = league_id * 5
    with open(project_path + "/res/leagues.metadata", "r") as file:
        csv_folder = file.readlines()[file_strt+3].rstrip()

    #2 sezone gleda za ime tima u data
    for i in range(2):
        csv = pd.read_csv("{}/res/{}/{}".format(project_path, csv_folder, os.listdir("{}/res/{}".format(project_path, csv_folder))[i]))
        if name in csv["HomeTeam"].array or name in csv["AwayTeam"].array:
            return name

    with open(project_path + "/res/premier_to_data_names", "r") as file:
        names = file.readlines()
    name = name + "\n"
    if name not in names:
        print("Data name for {}".format(name)); new_name = input()+"\n"
        names += [name, new_name]
        with open(project_path + "/res/premier_to_data_names", "w") as file:
            file.writelines(names)
        name = new_name
    else: name = names[names.index(name)+1]
    return name.rstrip()

class Date:
    def __init__(self, date_str):
        date_data = date_str.split("/")
        self.day = int(date_data[0])
        self.month = int(date_data[1])
        self.year = int(date_data[2])

    def day_difference(self, date):
        #no leap years, no different month day counts
        return self.day-date.day + 30*(self.month-date.month) + 365*(self.year-date.year)

class TeamStats:
    def __init__(self, ftg, s, st, c, f, r, y):
        self.ftg = ftg; self.s = s; self.st = st
        self.c = c; self.f = f; self.r = r; self.y = y

class MatchStats:
    def __init__(self, date_str, h_team, a_team, h_stats, a_stats, odds):
        self.hat = [h_team, a_team]
        self.h_stats = h_stats; self.a_stats = a_stats
        self.odds = odds
        self.date = Date(date_str=date_str)

    def result(self):
        if self.h_stats.ftg > self.a_stats.ftg: return "H"
        elif self.h_stats.ftg < self.a_stats.ftg: return "A"
        else: return "D"

    def winning_odds(self):
        return self.odds[('H', 'D', 'A').index(self.result())]

    def stats_for_team(self, T):
        return self.h_stats if T == self.hat[0] else self.a_stats

class LeagueSeason:
    def __init__(self, season, matches_stats):
        self.table_stats = {} #"TeamName" -> [pts, gd, ns]
        self.matches_stats = matches_stats
        self.season = season
        self.calculate_table()

    def calculate_table(self):
        for match_stats in self.matches_stats:
            h_pts_gd_ns = [0, 0, 0] if match_stats.hat[0] not in self.table_stats else self.table_stats[match_stats.hat[0]]
            a_pts_gd_ns = [0, 0, 0] if match_stats.hat[1] not in self.table_stats else self.table_stats[match_stats.hat[1]]
            h_stats = match_stats.h_stats; a_stats = match_stats.a_stats

            if h_stats.ftg > a_stats.ftg: h_pts_gd_ns[0] += 3
            elif h_stats.ftg < a_stats.ftg: a_pts_gd_ns[0] += 3
            else:
                h_pts_gd_ns[0] += 1
                a_pts_gd_ns[0] += 1

            h_pts_gd_ns[1] += (h_stats.ftg - a_stats.ftg)
            a_pts_gd_ns[1] += (a_stats.ftg - h_stats.ftg)
            h_pts_gd_ns[2] += 1
            a_pts_gd_ns[2] += 1

            self.table_stats[match_stats.hat[0]] = h_pts_gd_ns
            self.table_stats[match_stats.hat[1]] = a_pts_gd_ns

    def print_table(self):
        print(self.table_stats)

def K_ha(t, hat, what):
    if(t in hat):
        if t == what[0] and t == hat[0]: return k[8]
        if t == what[0] and t == hat[1]: return k[9]
        if t == what[1] and t == hat[1]: return k[10]
        if t == what[1] and t == hat[0]: return k[11]
    else: return (k[8]+k[9]+k[10]+k[11])/4

def M(t1, t2, hat):
    if t1 in hat and t2 in hat: return k[14]
    elif t1 in hat: return k[15]
    else: return 0  #Prepraviti paper

def D_sm(T, match_stats, algo_tactic):
    team_stats = match_stats.stats_for_team(T)
    r_ftg_st = team_stats.ftg/team_stats.st if team_stats.st!=0 else 0

    return [k[1] * team_stats.ftg,
            k[1]*team_stats.ftg*(1+k[4]*r_ftg_st),
            k[1]*team_stats.ftg + k[2]*team_stats.s + k[3]*team_stats.st + k[5]*team_stats.c + k[6]*team_stats.f - k[7]*(team_stats.r + 0.5*team_stats.y),
            k[1]*team_stats.ftg*(1+k[4]*r_ftg_st) + k[2]*team_stats.s + k[3]*team_stats.st + k[5]*team_stats.c + k[6]*team_stats.f - k[7]*(team_stats.r + 0.5*team_stats.y),
            k[1] * team_stats.ftg
            ][algo_tactic]

def ST_s(T, this_ssn, last_ssn):
    pts_gd_ns_1 = this_ssn.table_stats.get(T, (0,0,0))
    pts_gd_ns_2 = last_ssn.table_stats.get(T, (0,0,0))
    if pts_gd_ns_2[2] == 0: return 20 #paper
    s_ratio = k[13] * (pts_gd_ns_1[2] / pts_gd_ns_2[2])
    if s_ratio > 1: s_ratio = 1
    return s_ratio * (pts_gd_ns_1[0] + k[12] * pts_gd_ns_1[1]) + (1 - s_ratio) * (pts_gd_ns_2[0] + k[12] * pts_gd_ns_2[1])

def ST_smax(this_ssn, last_ssn):
    st_smax = 0
    for team in list(last_ssn.table_stats.keys()): # last season da se svi timovi vide ako neko u zadnjoj nije igrao niti jedan mec
        st_s = ST_s(team, this_ssn, last_ssn)
        if st_s > st_smax: st_smax = st_s
    return st_smax


def Strength(T, t1, t2, match_date, league_seasons, n_sall_max, algo_tactic):
    strength = 0
    n_sall = 0
    for league_season in league_seasons:
        if n_sall > n_sall_max: break
        for match_stats in league_season.matches_stats:
            m = M(T, t2 if t1==T else t1, match_stats.hat)
            day_diff = match_date.day_difference(match_stats.date)

            if m > 0 and day_diff > 0:
                n_sall += 1
                E = match_stats.hat[0] if T == match_stats.hat[1] else match_stats.hat[1]

                sts_max = ST_smax(league_seasons[0], league_seasons[1])
                STER_s = ST_s(E, league_seasons[0], league_seasons[1])/(sts_max)
                STTR_s = ST_s(T, league_seasons[0], league_seasons[1])/(sts_max)

                strength += (1 - math.exp(-k[0] / (n_sall*n_sall))) * m *\
                            (K_ha(T, (t1, t2), match_stats.hat)*D_sm(T, match_stats, algo_tactic) -
                             K_ha(E, (t1, t2), match_stats.hat)*D_sm(E, match_stats, algo_tactic))
            if n_sall>n_sall_max: break
    return strength

def SUPERIORITY(T, t1, t2, match_date, league_seasons, n_sall_max, algo_tactic):
    return Strength(T, t1, t2, match_date, league_seasons, n_sall_max, algo_tactic)/n_sall_max

def TP_hda(T, match_date, league_seasons, n_sall_max):
    tp_hda = [0,0,0]
    n_sall = 0
    for league_season in league_seasons:
        if n_sall > n_sall_max: break
        for match_stats in league_season.matches_stats:
            day_diff = match_date.day_difference(match_stats.date)
            if T in match_stats.hat and day_diff > 0:
                n_sall+=1
                result = match_stats.result()
                factor = (1 - math.exp(-k[16] / (n_sall*n_sall)))
                if T==match_stats.hat[0] and result=="H": tp_hda[0] += factor
                elif T==match_stats.hat[1] and result=="A": tp_hda[2] += factor
                else: tp_hda[1] += factor
            if n_sall>n_sall_max: break
    ftp_hda = sum(tp_hda)
    if ftp_hda==0: return [0.33, 0.33, 0.33]
    return list(v/ftp_hda for v in tp_hda)

def B(T, i, league_season):
    table_stats = league_season.table_stats

    if len(list(table_stats.keys())) > 4 and table_stats.get(T) is not None:
        table_list = [(key, value) for key, value in sorted(table_stats.items(), key=lambda item: item[1][0], reverse=True)]

        for k in range(len(table_list)):
            if table_list[k][0] == T:
                if k-i < 0 or k-i >= len(table_list):
                    return 0.125
                b_diff = math.fabs(math.copysign(1, i)*(table_list[k-i][1][0] - table_list[k][1][0]))
                if b_diff < 3: return 0.25
                elif b_diff == 3: return 0.15
                else: return 0 #popraviti u paperu
    return 0.5 # popraviti u paperu, tek pocela nova sezona tako da mu je table pressure oko 0.5

def RZ(t1, t2, league_season):
    t1b = 0; t2b = 0
    if len(list(league_season.table_stats.keys())) > 5 and (league_season.table_stats.get(t1) is not None and league_season.table_stats.get(t2) is not None):
        t1b = league_season.table_stats[t1][0]
        t2b = league_season.table_stats[t2][0]
    if t1b == 0 and t2b == 0: return 1
    return 1-(t1b-t2b)/max(t1b, t2b) if(t1b >= t2b) else 1-k[20]*(t1b-t2b)/max(t1b, t2b)

def WILL(T, t1, t2, match_date, sup1, sup2, league_seasons, n_sall_max):
    tp_hda = TP_hda(T, match_date, league_seasons, n_sall_max)
    side_wd = (tp_hda[0 if T==t1 else 2] + 0.5*tp_hda[1])
    league_pts_diff = RZ(t1, t2, league_seasons[0])
    table_pressure = 0
    for i in range(-2, 3):
        if i!=0: table_pressure += B(T, i, league_seasons[0])
    return (sup1+sup2)/2 * (k[17]*side_wd + k[18]*table_pressure + k[19]*league_pts_diff)

def stripped_seasons(match_date, league_seasons): # ordered seasons in list
    pop_n_seasons = 0
    for league_season in league_seasons:
        match_in_season = False
        for match_stats in league_season.matches_stats:
            day_diff = match_date.day_difference(match_stats.date)
            if day_diff >= 0:
                match_in_season = True
                break
        if not match_in_season: pop_n_seasons += 1
        else: break
    for i in range(pop_n_seasons):
        league_seasons.pop(0)

def MTD(T1, T2, match_date, league_seasons, n_sall_max, algo_tactic):
    sup1 = SUPERIORITY(T1, T1, T2, match_date, league_seasons, n_sall_max, algo_tactic)
    sup2 = SUPERIORITY(T2, T1, T2, match_date, league_seasons, n_sall_max, algo_tactic)
    will1 = 0; will2 = 0
    if algo_tactic != 4:
        will1 = WILL(T1, T1, T2, match_date, sup1, sup2, league_seasons, n_sall_max)
        will2 = WILL(T2, T1, T2, match_date, sup1, sup2, league_seasons, n_sall_max)
    print(sup1, will1, sup2, will2)
    return sup1 + will1 - (sup2 + will2)

def LMS(up_until_date, league_seasons, n_seasons_mtd, matches_per_season, algo_tactic):
    lms = []
    n_sall_max = n_seasons_mtd * matches_per_season
    #strp_seasons = league_seasons.copy()
    #stripped_seasons(match_date, strp_seasons)

    for league_season in league_seasons:
        for match_stats in league_season.matches_stats:
            if up_until_date.day_difference(match_stats.date) > 0:
                lms.append((MTD(match_stats.hat[0], match_stats.hat[1], match_stats.date, league_seasons, n_sall_max, algo_tactic),
                            match_stats.result()))

    print("NUMBER OF MATCHES: ", len(lms), " for league seasons: ", len(league_seasons))

    # matches = list(map(lambda item: item[0], lms))
    # home = list(map(lambda item: item[0], filter(lambda item: item[1]=='H', lms)))
    # away = list(map(lambda item: item[0], filter(lambda item: item[1]=='A', lms)))
    # draw = list(map(lambda item: item[0], filter(lambda item: item[1] == 'D', lms)))
    #
    # print("SUM: ", sum(matches))
    #
    # plt.hist([matches, home, away, draw], bins=50, label=['Matches', 'Home', 'Away', 'Draw'])
    # plt.title("League {} stats for last {} seasons".format(league_seasons[0].season.split(" ")[0], len(league_seasons)))
    # plt.xlabel('MTD')
    # plt.legend(loc='upper right')
    # plt.show()

    return lms

def get_league_seasons(league_id, n_seasons_lms):
    file_strt = league_id * 5
    with open(project_path + "/res/leagues.metadata", "r") as file:
        csv_folder = file.readlines()[file_strt + 3].rstrip()

    league_seasons = []
    league_seasons_csv = sorted(os.listdir("{}/res/{}".format(project_path, csv_folder)), key=lambda item: item, reverse=True)
    for i in range(n_seasons_lms):
        csv = pd.read_csv("{}/res/{}/{}".format(project_path, csv_folder, league_seasons_csv[i]))

        match_stats = []
        dates = csv["Date"].array
        h_teams = csv["HomeTeam"].array
        a_teams = csv["AwayTeam"].array

        ftg = [csv["FTHG"].array, csv["FTAG"].array]
        s = [csv["HS"].array, csv["AS"].array]
        st = [csv["HST"].array, csv["AST"].array]
        c = [csv["HC"].array, csv["AC"].array]
        f = [csv["HF"].array, csv["AF"].array]
        r = [csv["HR"].array, csv["AR"].array]
        y = [csv["HY"].array, csv["AY"].array]
        odds = [csv["B365H"].array, csv["B365D"].array, csv["B365A"].array]

        for k in range(len(dates)):
            match_stats.append(MatchStats(
                dates[k], h_teams[k], a_teams[k],
                TeamStats(ftg[0][k], s[0][k], st[0][k], c[0][k], f[0][k], r[0][k], y[0][k]),
                TeamStats(ftg[1][k], s[1][k], st[1][k], c[1][k], f[1][k], r[1][k], y[1][k]),
                (odds[0][k], odds[1][k], odds[2][k])
            ))

        league_seasons.append(LeagueSeason("{} {}".format(csv_folder, dates[-1].split("/")[-1]), match_stats))

    return league_seasons

def season_money_yield(epsilon, bet, up_until_date, season_index, league_seasons, n_seasons_mtd, matches_per_season, money_yield_tactics, algo_tactics):
    balance = []
    r = ('H', 'D', 'A')
    n_sall_max = n_seasons_mtd * matches_per_season

    for k in range(len(algo_tactics)):
        lms = LMS(up_until_date, league_seasons, n_seasons_mtd, matches_per_season, algo_tactics[k])
        for i in range(len(money_yield_tactics)):
            balance.append([0])

        for match_stats in league_seasons[season_index].matches_stats:
            mtd = MTD(match_stats.hat[0], match_stats.hat[1], match_stats.date, league_seasons, n_sall_max,
                      algo_tactics[k])
            hda = [0, 0, 0]
            for lm in lms:
                if math.fabs(mtd - lm[0]) < epsilon:
                    if lm[1] == 'H':
                        hda[0] += 1
                    elif lm[1] == 'A':
                        hda[2] += 1
                    else:
                        hda[1] += 1

            fhda = sum(hda)
            print("{} matches around {} MTD with epsilon gap {}".format(fhda, mtd, epsilon))
            if fhda == 0: continue
            calc_perc_odds = (hda[0]/fhda, hda[1]/fhda, hda[2]/fhda)
            their_perc_odds = [1/match_stats.odds[0], 1/match_stats.odds[1], 1/match_stats.odds[2]]

            their_max_perc_odds = max(their_perc_odds)
            calc_odd_for_their_max = calc_perc_odds[their_perc_odds.index(their_max_perc_odds)]
            lenmyt = len(money_yield_tactics)

            for i in range(lenmyt):
                max_perc_odd_diff = calc_odd_for_their_max - their_max_perc_odds
                draw_odd = calc_perc_odds[1] - min(calc_perc_odds)

                # value bets (bet variation) v2
                # max_calc_odd = max(calc_perc_odds)
                # if money_yield_tactics[i] == 1:
                #     print("home")
                #     new_bet = ((calc_perc_odds[0]-min(calc_perc_odds))*4+1)*bet
                #     balance[-(lenmyt - i)].append(
                #         balance[-(lenmyt - i)][-1] + (new_bet * (match_stats.winning_odds() - 1)
                #             if 'H' == match_stats.result() else -new_bet)
                #     )
                # continue
                # value bets
                print("Their odds ",their_perc_odds, " , calculated odds ", calc_perc_odds, " , odd diff ",max_perc_odd_diff)
                if money_yield_tactics[i] == 0:
                    print("Result : ", match_stats.result(), " , predicted result: ", r[hda.index(
                            max(hda))])
                    balance[-(lenmyt-i)].append(
                        balance[-(lenmyt-i)][-1] + (bet * (match_stats.winning_odds() - 1) if r[hda.index(
                            max(hda))] == match_stats.result() else -bet)
                    )
                #value bets (bet variation)
                elif money_yield_tactics[i] == 1:
                    print("Result : ", match_stats.result(), " , predicted result: ", r[hda.index(
                        max(hda))])
                    if (2+10*max_perc_odd_diff)*bet>=1:
                        balance[-(lenmyt-i)].append(
                            balance[-(lenmyt-i)][-1] + ((2+10*max_perc_odd_diff)*bet * (match_stats.winning_odds() - 1) if r[hda.index(
                                max(hda))] == match_stats.result() else -((2+10*max_perc_odd_diff)*bet))
                        )
                #draw bets (bet variation)
                elif money_yield_tactics[i] == 2 and draw_odd>0:
                    print("Result : ", match_stats.result(), " , predicted result: ", "D")
                    balance[-(lenmyt-i)].append(
                        balance[-(lenmyt-i)][-1] + ((bet+draw_odd*10) * (match_stats.winning_odds() - 1)
                        if 'D' == match_stats.result() else -(bet+draw_odd*10))
                    )

    #colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange')

    NUM_COLORS = len(algo_tactics)*len(money_yield_tactics)

    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    t = 0
    for k in range(len(algo_tactics)):
        for i in range(len(money_yield_tactics)):
            label = "myt={} at={}".format(money_yield_tactics[i], algo_tactics[k])
            print("EARNED {} KM for myt={} at={}".format(balance[t][-1], money_yield_tactics[i], algo_tactics[k]))
            ax.plot(balance[t], label=label)
            t+=1

    plt.legend(loc='best')
    plt.title("{} for ~{}KM bet/match".format(league_seasons[season_index].season, bet))
    plt.ylabel('Balance')
    plt.xlabel('Matches')
    plt.show()



#INFO SA PREMIERA
league_name = "NJEMAČKA 1.LIGA"
home = "Bayern Munich"
away = "Fortuna Dusseldorf"
match_date = Date("30/05/2020")
h = 4.4; d = 4.2; a = 1.7
#...
#INFO SA PREMIERA

matches_per_season = 38
n_seasons_lms = 8
n_seasons_mtd = 2


driver = webdriver.Chrome(executable_path=project_path + "/res/chromedriver")

update_info = update_league_data(league_id=leagues_id[league_name], n_last_seasons=n_seasons_lms+n_seasons_mtd)
if update_info["updated"]:
    print("League {} updated at {} !".format(league_name, update_info["text"].split(" ")[2]))
else:
    print("League {} was already up to date {} !".format(league_name, update_info["text"].split(" ")[2]))

home = premier_to_data_team_name(name=home, league_id=leagues_id[league_name])
away = premier_to_data_team_name(name=away, league_id=leagues_id[league_name])

driver.close()



league_seasons = get_league_seasons(leagues_id[league_name], n_seasons_lms+n_seasons_mtd)

# 0 pure_goal_diff
# 1 goal_diff_target_shots
# 2 all_stats_pure_goal_diff
# 3 all_stats_goal_diff_target_shots
# 4 pure_goal_diff_no_will
algo_tactics = [0,1,3]

# 0 value bets
# 1 value bets (bet variation)
# 2 draw bets (bet variation)
money_yield_tactics = [0,1]

#print(MTD(home, away, match_date, league_seasons, n_seasons_mtd*matches_per_season))
season_money_yield(5, 1, Date("18/08/17"), 2, league_seasons, n_seasons_mtd, matches_per_season, money_yield_tactics, algo_tactics)
#LMS(match_date, league_seasons, n_seasons_mtd*matches_per_season)

# BET 25KM
# EARNED 493.25 KM for myt=0 at=0
# EARNED 857.5986448719767 KM for myt=1 at=0
# EARNED -243.61211529598577 KM for myt=2 at=0
# EARNED 491.5 KM for myt=0 at=1
# EARNED 939.563775767964 KM for myt=1 at=1
# EARNED -189.58342326265824 KM for myt=2 at=1
# EARNED 325.5 KM for myt=0 at=2
# EARNED 649.0695730950385 KM for myt=1 at=2
# EARNED 232.17254174810708 KM for myt=2 at=2
# EARNED 320.5 KM for myt=0 at=3
# EARNED 787.5844948466096 KM for myt=1 at=3
# EARNED 7.788084401222736 KM for myt=2 at=3
# EARNED 359.0 KM for myt=0 at=4
# EARNED 691.1542467545335 KM for myt=1 at=4
# EARNED -217.88506386152076 KM for myt=2 at=4