import random
import numpy as np
import pymc as pm
import arviz as az
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import threading

###############################################################################
# 1) Dati statici dei 10 giocatori (caratteristiche fisse)
###############################################################################
players_data = [
    {"name": "Alex",   "anni_di_esperienza": 2,  "partite_vinte": 5,   "partite_perse": 3,  "form_index": 7},
    {"name": "Bruno",  "anni_di_esperienza": 3,  "partite_vinte": 7,   "partite_perse": 2,  "form_index": 6},
    {"name": "Carlo",  "anni_di_esperienza": 5,  "partite_vinte": 10,  "partite_perse": 5,  "form_index": 8},
    {"name": "Dario",  "anni_di_esperienza": 1,  "partite_vinte": 3,   "partite_perse": 6,  "form_index": 5},
    {"name": "Enea",   "anni_di_esperienza": 4,  "partite_vinte": 8,   "partite_perse": 3,  "form_index": 7},
    {"name": "Fabio",  "anni_di_esperienza": 6,  "partite_vinte": 15,  "partite_perse": 9,  "form_index": 9},
    {"name": "Gino",   "anni_di_esperienza": 2,  "partite_vinte": 4,   "partite_perse": 7,  "form_index": 4},
    {"name": "Marco",  "anni_di_esperienza": 8,  "partite_vinte": 20,  "partite_perse": 10, "form_index": 10},
    {"name": "Paolo",  "anni_di_esperienza": 5,  "partite_vinte": 12,  "partite_perse": 6,  "form_index": 7},
    {"name": "Rocco",  "anni_di_esperienza": 2,  "partite_vinte": 4,   "partite_perse": 4,  "form_index": 5},
]

# Mischiamo i 10 giocatori in modo casuale
random.shuffle(players_data)

###############################################################################
# 2) Seleziona i 2 capitani e crea le due squadre
###############################################################################
two_captains = random.sample(players_data, 2)
captain1, captain2 = two_captains[0], two_captains[1]

sorted_names = sorted([captain1["name"], captain2["name"]])
if sorted_names[0] == captain1["name"]:
    first_captain, second_captain = captain1, captain2
else:
    first_captain, second_captain = captain2, captain1

remaining_players = [p for p in players_data if p not in [captain1, captain2]]
team_first = [first_captain]
team_second = [second_captain]

###############################################################################
# 2b) Definiamo la preferenza "reale" di ciascun capitano
###############################################################################
# Possibili preferenze: "esperienza", "vittorie", "forma", "perdite"
captain_prefs = ["esperienza", "vittorie", "forma", "perdite"]
cap1_pref = random.choice(captain_prefs)
cap2_pref = random.choice(captain_prefs)

# K grande, es. K=15.0 => effetto molto evidente
K = 15.0

###############################################################################
# 3) Liste globali per tracciare l'andamento delle probabilità
###############################################################################
bayes_history = []
classic_history = []
chosen_history = []  # giocatori scelti (per "insegnare" al bayes come ragionare)

###############################################################################
# Creazione GUI
###############################################################################
root = tk.Tk()
root.title("Selezione Squadre - Distribuzione di Probabilità (Bayes vs Classica)")
root.geometry("1300x900")
root.configure(bg="#F0F0F0")  # sfondo grigio chiaro

style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background="#F0F0F0")
style.configure("TLabel", background="#F0F0F0", font=("Arial", 14))
style.configure("TButton", font=("Arial", 14, "bold"), padding=10)

TITLE_FONT = ("Arial", 18, "bold")
BIG_FONT = ("Arial", 16)
NORMAL_FONT = ("Arial", 14)
SMALL_FONT = ("Arial", 12)

pool = remaining_players[:]

# FRAME in alto
top_frame = ttk.Frame(root, style="TFrame")
top_frame.pack(pady=10)

title_label = ttk.Label(
    top_frame,
    text="Selezione Squadre (Distribuzione di Probabilità su tutti i giocatori)",
    style="TLabel"
)
title_label.config(font=TITLE_FONT)
title_label.pack()

info_label = ttk.Label(top_frame, text="Premi i pulsanti per calcolare e scegliere", style="TLabel")
info_label.config(font=BIG_FONT)
info_label.pack(pady=5)

prob_label = ttk.Label(top_frame, text="", style="TLabel")
prob_label.config(font=BIG_FONT, foreground="blue")
prob_label.pack(pady=5)

# Label preferenze dei capitani
pref_label = ttk.Label(top_frame, text="", style="TLabel")
pref_label.config(font=BIG_FONT, foreground="purple")
pref_label.pack(pady=5)

def param_string(pref):
    if pref=="esperienza":
        return "Esperienza"
    elif pref=="vittorie":
        return "Vittorie"
    elif pref=="forma":
        return "Forma"
    elif pref=="perdite":
        return "Perdite (penalizzazione maggiore)"
    else:
        return pref

pref_text = (
    f"Capitano {first_captain['name']} privilegia: {param_string(cap1_pref)} (K={K})\n"
    f"Capitano {second_captain['name']} privilegia: {param_string(cap2_pref)} (K={K})"
)
pref_label.config(text=pref_text)

progress = ttk.Progressbar(top_frame, orient='horizontal', length=400, mode='indeterminate')
progress.pack(pady=5)
progress.pack_forget()

# Label per mostrare CHI sceglie la prossima volta
next_captain_label = ttk.Label(top_frame, text="", style="TLabel")
next_captain_label.config(font=BIG_FONT, foreground="red")
next_captain_label.pack(pady=5)

###############################################################################
# Frame con pulsanti
###############################################################################
button_frame = ttk.Frame(root, style="TFrame")
button_frame.pack(pady=10)

calc_prob_button = ttk.Button(button_frame, text="Calcola Probabilità (Prossimo Pick)")
calc_prob_button.grid(row=0, column=0, padx=20)

choice_button = ttk.Button(button_frame, text="Scelta (non bayesiana)")
choice_button.grid(row=0, column=1, padx=20)

stats_bayes_button = ttk.Button(button_frame, text="Stampa statistiche bayesiane")
stats_bayes_button.grid(row=1, column=0, pady=10)

stats_classic_button = ttk.Button(button_frame, text="Stampa statistiche classiche")
stats_classic_button.grid(row=1, column=1, pady=10)

###############################################################################
# Frame giocatori
###############################################################################
players_frame = ttk.Frame(root, style="TFrame")
players_frame.pack(pady=10)

player_labels = []

def create_player_labels():
    for i, player in enumerate(players_data):
        lbl = ttk.Label(players_frame, text="", style="TLabel")
        lbl.config(font=NORMAL_FONT, width=50, background="white", borderwidth=2, relief="groove")
        lbl.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="nsew")
        player_labels.append(lbl)

create_player_labels()

def refresh_player_labels():
    for lbl, player in zip(player_labels, players_data):
        txt = (f"{player['name']} "
               f"(Exp={player['anni_di_esperienza']}, Win={player['partite_vinte']}, "
               f"Lost={player['partite_perse']}, Form={player['form_index']})")
        lbl.config(text=txt, foreground="black", background="white")
    
    # Capitani colorati
    for lbl, player in zip(player_labels, players_data):
        if player == first_captain:
            lbl.config(foreground="red")
        elif player == second_captain:
            lbl.config(foreground="green")
    
    # Già scelti (tranne i capitani)
    chosen_players = team_first + team_second
    for lbl, player in zip(player_labels, players_data):
        if player in chosen_players and player not in [first_captain, second_captain]:
            lbl.config(foreground="gray")

refresh_player_labels()

def get_current_captain_and_team():
    # pick "attuale"
    picks_done = len(team_first) + len(team_second) - 2
    if picks_done % 2 == 0:
        return first_captain, team_first, cap1_pref
    else:
        return second_captain, team_second, cap2_pref

def get_next_captain():
    # prossimo pick => picks_done+1
    picks_done = len(team_first) + len(team_second) - 2
    next_picks_done = picks_done + 1
    if next_picks_done % 2 == 0:
        return first_captain
    else:
        return second_captain

def update_next_captain_label():
    if not pool:
        next_captain_label.config(text="Tutti i giocatori scelti, nessun capitano successivo.")
        return
    # Calcoliamo chi sceglierà al prossimo giro
    cpt = get_next_captain()
    next_captain_label.config(text=f"Prossima scelta: {cpt['name']}")

###############################################################################
# synergy invariato
###############################################################################
def compute_synergy(player, chosen_players):
    if not chosen_players:
        return 0.0
    xp_vals = [p["anni_di_esperienza"] for p in chosen_players]
    win_vals = [p["partite_vinte"] for p in chosen_players]
    form_vals = [p["form_index"] for p in chosen_players]
    
    xp_mean = np.mean(xp_vals)
    win_mean = np.mean(win_vals)
    form_mean = np.mean(form_vals)
    
    dist = abs(player["anni_di_esperienza"] - xp_mean)
    dist += abs(player["partite_vinte"] - win_mean)
    dist += abs(player["form_index"] - form_mean)
    
    return -dist

def guess_preference_from_choices():
    if len(chosen_history) == 0:
        return {
            "xp_mu": 1.0,
            "win_mu": 2.0,
            "loss_mu": -1.0,
            "form_mu": 1.0
        }
    
    xp_mean = np.mean([p["anni_di_esperienza"] for p in chosen_history])
    win_mean = np.mean([p["partite_vinte"] for p in chosen_history])
    form_mean = np.mean([p["form_index"] for p in chosen_history])
    inv_loss_vals = [100 - p["partite_perse"] for p in chosen_history]
    inv_loss_mean = np.mean(inv_loss_vals)
    
    vals = {
        "xp_mean": xp_mean,
        "win_mean": win_mean,
        "inv_loss_mean": inv_loss_mean,
        "form_mean": form_mean
    }
    best_param = max(vals, key=vals.get)
    
    xp_mu = 1.0
    win_mu = 2.0
    loss_mu = -1.0
    form_mu = 1.0
    
    if best_param == "xp_mean":
        xp_mu = 3.0
    elif best_param == "win_mean":
        win_mu = 4.0
    elif best_param == "inv_loss_mean":
        loss_mu = -2.0
    elif best_param == "form_mean":
        form_mu = 3.0
    
    return {
        "xp_mu": xp_mu,
        "win_mu": win_mu,
        "loss_mu": loss_mu,
        "form_mu": form_mu
    }

def build_bayesian_model_and_sample_all(players_pool):
    """
    Calcola la distribuzione di probabilità (Bayes) che ciascun giocatore in players_pool 
    venga scelto al prossimo pick, analizzando le caratteristiche dei giocatori già scelti 
    **nella squadra del capitano che sta per scegliere** (current_team) e quelle degli 
    elementi ancora nel pool.
    I prior per beta_xp, beta_win, beta_loss e beta_form sono adattati (tramite
    guess_preference_from_choices) in base alle scelte passate (globali), ma per il calcolo
    della "synergy" si considerano solo i giocatori nella squadra del capitano corrente.
    """
    if not players_pool:
        return []
    
    # Otteniamo il capitano che sta per scegliere e la sua squadra (current_team)
    current_cap, current_team, current_pref = get_current_captain_and_team()
    
    # Per la "synergy" consideriamo solo i giocatori già scelti nella squadra del capitano corrente
    chosen_current = current_team  # Esclude gli elementi dell'altra squadra
    
    # Usiamo la funzione guess_preference_from_choices() per ottenere i prior da usare
    # (questa funzione analizza le scelte passate, come da versione originale)
    prefs = guess_preference_from_choices()
    
    # Prepariamo le liste con le caratteristiche per ogni giocatore nel pool
    xp_list = []
    wins_list = []
    losses_list = []
    form_list = []
    synergy_list = []
    
    for pl in players_pool:
        xp_list.append(pl["anni_di_esperienza"])
        wins_list.append(pl["partite_vinte"])
        losses_list.append(pl["partite_perse"])
        form_list.append(pl["form_index"])
        # Qui la synergy viene calcolata **solo** in base ai giocatori della squadra del capitano corrente
        synergy_list.append(compute_synergy(pl, chosen_current))
    
    xp_arr = np.array(xp_list)
    win_arr = np.array(wins_list)
    loss_arr = np.array(losses_list)
    form_arr = np.array(form_list)
    syn_arr = np.array(synergy_list)
    
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta_xp = pm.Normal("beta_xp", mu=prefs["xp_mu"], sigma=1)
        beta_win = pm.Normal("beta_win", mu=prefs["win_mu"], sigma=1)
        beta_loss = pm.Normal("beta_loss", mu=prefs["loss_mu"], sigma=1)
        beta_form = pm.Normal("beta_form", mu=prefs["form_mu"], sigma=1)
        beta_syn = pm.Normal("beta_syn", mu=1, sigma=1)
        
        score = (alpha
                 + beta_xp * xp_arr
                 + beta_win * win_arr
                 + beta_loss * loss_arr
                 + beta_form * form_arr
                 + beta_syn * syn_arr)
        
        probs = pm.Deterministic("probs", pm.math.softmax(score))
        
        trace = pm.sample(draws=400, tune=200, chains=1, progressbar=False, random_seed=42)
    
    posterior_probs = trace.posterior["probs"].values[0]
    mean_probs = posterior_probs.mean(axis=0)
    return mean_probs

def compute_classic_probabilities(players_pool):
    if not players_pool:
        return []
    
    scores = []
    for p in players_pool:
        base_score = (2.0 * p["partite_vinte"]
                      + p["anni_di_esperienza"]
                      + p["form_index"]
                      - 0.5 * p["partite_perse"])
        
        if base_score <= 0:
            extra = 0.0
        else:
            max_extra = 0.2 * base_score
            #extra = random.uniform(0, max_extra) commentato per togliere la parte di casualità
            extra = 0 
        total_score = base_score + extra
        scores.append(total_score)
    
    sum_scores = sum(scores)
    if sum_scores <= 0:
        return [1.0 / len(players_pool)] * len(players_pool)
    
    return [s / sum_scores for s in scores]

def pick_next_player_non_bayesian(players_pool):
    if len(players_pool) == 1:
        return players_pool[0]
    
    # Leggiamo la preferenza del capitano
    current_captain, current_team, current_pref = get_current_captain_and_team()
    
    best_score = float('-inf')
    best_player = None
    
    for p in players_pool:
        # Punteggio base
        base_score = (2.0 * p["partite_vinte"]
                      + p["anni_di_esperienza"]
                      + p["form_index"]
                      - 0.5 * p["partite_perse"])
        
        # Applichiamo K=9 se preferito
        if current_pref == "vittorie":
            delta = (K - 1.0)*(2.0 * p["partite_vinte"])
            base_score += delta
        elif current_pref == "esperienza":
            delta = (K - 1.0)*(p["anni_di_esperienza"])
            base_score += delta
        elif current_pref == "forma":
            delta = (K - 1.0)*(p["form_index"])
            base_score += delta
        elif current_pref == "perdite":
            # penalizziamo ancor di più le partite perse
            delta = (K - 1.0)*(-0.5 * p["partite_perse"])
            base_score += delta
        
        # Componente random
        if base_score <= 0:
            extra = 0.0
        else:
            max_extra = 0.2 * base_score
            extra = random.uniform(0, max_extra)
        
        final_score = base_score + extra
        if final_score > best_score:
            best_score = final_score
            best_player = p
    
    return best_player

def do_calcola_prob():
    if not pool:
        info_label.config(text="Tutti i giocatori sono stati scelti. Non c'è più nessuno nel pool.")
        return
    
    progress.pack()
    progress.start(10)
    
    def background_calc():
        bayes_probs = build_bayesian_model_and_sample_all(pool)
        classic_probs = compute_classic_probabilities(pool)
        
        bayes_vector = [0.0]*len(players_data)
        classic_vector = [0.0]*len(players_data)
        
        for i, p in enumerate(players_data):
            if p in pool:
                j = pool.index(p)
                bayes_vector[i] = bayes_probs[j]
                classic_vector[i] = classic_probs[j]
            else:
                bayes_vector[i] = 0.0
                classic_vector[i] = 0.0
        
        bayes_history.append(bayes_vector)
        classic_history.append(classic_vector)
        
        def on_done():
            progress.stop()
            progress.pack_forget()
            
            info_label.config(text="Distribuzione di probabilità: Bayes vs Classico (prossimo pick).")
            
            lines = []
            lines.append("PROBABILITÀ CALCOLATE SUI GIOCATORI NEL POOL:\n")
            for j, pl in enumerate(pool):
                lines.append(f"{pl['name']}: Bayes={bayes_probs[j]:.3f}, Classico={classic_probs[j]:.3f}")
            text_out = "\n".join(lines)
            prob_label.config(text=text_out)
            
            # Grafico a barre di confronto
            x_labels = [p["name"] for p in pool]
            x = np.arange(len(pool))
            
            plt.figure()
            plt.title("Confronto: Bayes vs Classico (Prossimo Pick)")
            
            width = 0.35
            plt.bar(x - width/2, bayes_probs, width=width, label="Bayes")
            plt.bar(x + width/2, classic_probs, width=width, label="Classico")
            
            plt.xticks(x, x_labels, rotation=45, ha="right")
            plt.ylabel("Probabilità")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        root.after(0, on_done)
    
    threading.Thread(target=background_calc).start()

def do_scelta():
    global pool
    if not pool:
        info_label.config(text="Non ci sono più giocatori disponibili.")
        calc_prob_button.config(state="disabled")
        choice_button.config(state="disabled")
        return
    
    current_captain, current_team, current_pref = get_current_captain_and_team()
    picks_done = len(team_first) + len(team_second) - 2
    
    chosen_one = pick_next_player_non_bayesian(pool)
    current_team.append(chosen_one)
    pool.remove(chosen_one)
    
    # Salviamo la scelta
    chosen_history.append(chosen_one)
    
    info_label.config(
        text=f"Pick #{picks_done+1} - Capitano {current_captain['name']} ha scelto: {chosen_one['name']}"
    )
    refresh_player_labels()
    
    if not pool:
        info_label.config(text="Tutti i giocatori sono stati scelti. Fine selezioni.")
        calc_prob_button.config(state="disabled")
        choice_button.config(state="disabled")
    else:
        # Aggiorniamo la label con il prossimo capitano
        update_next_captain_label()

def do_stats_bayes():
    if len(bayes_history) == 0:
        info_label.config(text="Nessun dato bayesiano salvato (non hai mai premuto 'Calcola Probabilità'?).")
        return
    
    n_turns = len(bayes_history)
    x_values = range(1, n_turns+1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Andamento Probabilità Bayesiane (per giocatore)")
    
    for i in range(len(players_data)):
        y = [bayes_history[t][i] for t in range(n_turns)]
        player_name = players_data[i]["name"]
        plt.plot(x_values, y, label=player_name)
    
    plt.xlabel("Turno # (click Calcola Probabilità)")
    plt.ylabel("Prob. Bayesiana")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def do_stats_classic():
    if len(classic_history) == 0:
        info_label.config(text="Nessun dato classico salvato (non hai mai premuto 'Calcola Probabilità'?).")
        return
    
    n_turns = len(classic_history)
    x_values = range(1, n_turns+1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Andamento Probabilità Classiche (per giocatore)")
    
    for i in range(len(players_data)):
        y = [classic_history[t][i] for t in range(n_turns)]
        player_name = players_data[i]["name"]
        plt.plot(x_values, y, label=player_name)
    
    plt.xlabel("Turno # (click Calcola Probabilità)")
    plt.ylabel("Prob. Classica (score normalizzato)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

calc_prob_button.config(command=do_calcola_prob)
choice_button.config(command=do_scelta)
stats_bayes_button.config(command=do_stats_bayes)
stats_classic_button.config(command=do_stats_classic)

# Definiamo e richiamiamo la update_next_captain_label iniziale
next_captain_label = ttk.Label(top_frame, text="", style="TLabel", foreground="red", font=BIG_FONT)
next_captain_label.pack(pady=5)

def update_next_captain_label():
    if not pool:
        next_captain_label.config(text="Tutti i giocatori sono stati scelti.")
        return
    picks_done = len(team_first) + len(team_second) - 2
    # prossimo pick => picks_done+1
    next_picks_done = picks_done + 1
    if next_picks_done % 2 == 0:
        cpt = second_captain
    else:
        cpt = first_captain
    next_captain_label.config(text=f"Prossima scelta: {cpt['name']}")

# Mostriamo subito chi sceglie per primo
update_next_captain_label()

root.mainloop()
