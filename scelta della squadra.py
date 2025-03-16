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

# Mischiamo l'ordine in modo casuale
random.shuffle(players_data)

###############################################################################
# 2) Scelta dei due capitani e creazione squadre
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
# 3) Liste globali per salvare l'andamento delle probabilità
#    bayes_history[t][i] => probabilità bayesiana di i-esimo giocatore a "turno t"
#    classic_history[t][i] => idem per probabilità classica
###############################################################################
bayes_history = []
classic_history = []

###############################################################################
# 4) Creazione GUI (finestra e stili)
###############################################################################
root = tk.Tk()
root.title("Selezione Squadre - Distribuzione di Probabilità (Bayes vs Classica)")
root.geometry("1300x900")
root.configure(bg="#F0F0F0")  # sfondo grigio chiaro

# Stile ttk "moderno"
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

###############################################################################
# Frame superiore con etichette e progress bar
###############################################################################
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

progress = ttk.Progressbar(top_frame, orient='horizontal', length=400, mode='indeterminate')
progress.pack(pady=5)
progress.pack_forget()

###############################################################################
# Frame pulsanti
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

###############################################################################
# Funzioni di refresh
###############################################################################
def refresh_player_labels():
    for lbl, player in zip(player_labels, players_data):
        txt = (f"{player['name']} "
               f"(Exp={player['anni_di_esperienza']}, Win={player['partite_vinte']}, "
               f"Lost={player['partite_perse']}, Form={player['form_index']})")
        lbl.config(text=txt, foreground="black", background="white")
    
    # Capitani
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
    picks_done = len(team_first) + len(team_second) - 2
    if picks_done % 2 == 0:
        return first_captain, team_first
    else:
        return second_captain, team_second

###############################################################################
# Calcolo synergy: gioca su xp, wins, form => -distanza
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
    
    return -dist  # synergy = - distanza

###############################################################################
# 1) build_bayesian_model_and_sample_all: restituisce TUTTE le probabilità (Bayes)
###############################################################################
def build_bayesian_model_and_sample_all(players_pool):
    """
    Restituisce un vettore di probabilità (Bayes) che ciascun
    giocatore in players_pool sia scelto ADESSO (prossimo pick).
    Usa:
      - xp, wins, losses, form
      - synergy (rispetto a TUTTI i giocatori già scelti)
    """
    if not players_pool:
        return []
    
    chosen_global = team_first + team_second
    
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
        synergy_list.append(compute_synergy(pl, chosen_global))
    
    xp_arr = np.array(xp_list)
    win_arr = np.array(wins_list)
    loss_arr = np.array(losses_list)
    form_arr = np.array(form_list)
    syn_arr = np.array(synergy_list)
    
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        
        beta_xp = pm.Normal("beta_xp", mu=1, sigma=1)
        beta_win = pm.Normal("beta_win", mu=2, sigma=1)
        beta_loss = pm.Normal("beta_loss", mu=-1, sigma=1)
        beta_form = pm.Normal("beta_form", mu=1, sigma=1)
        beta_syn = pm.Normal("beta_syn", mu=1, sigma=1)
        
        score = (alpha
                 + beta_xp * xp_arr
                 + beta_win * win_arr
                 + beta_loss * loss_arr
                 + beta_form * form_arr
                 + beta_syn * syn_arr)
        
        probs = pm.Deterministic("probs", pm.math.softmax(score))
        
        trace = pm.sample(draws=400, tune=200, chains=1, progressbar=False, random_seed=42)
    
    posterior_probs = trace.posterior["probs"].values[0]  # shape = [n_draws, pool_size]
    mean_probs = posterior_probs.mean(axis=0)             # shape = [pool_size,]
    return mean_probs

###############################################################################
# 2) Calcolo "probabilità classiche" da uno score non bayesiano
###############################################################################
def compute_classic_probabilities(players_pool):
    """
    Calcola uno score deterministico per ogni giocatore, poi normalizza
    per ottenere una "probabilità classica".
    Lo score base:  2×win + xp + form - 0.5×lost
    Poi la parte random = random(0, max_extra)
    dove max_extra = 0.2 * (score base)  se score base > 0, altrimenti 0
    """
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
            extra = random.uniform(0, max_extra)
        
        total_score = base_score + extra
        scores.append(total_score)
    
    sum_scores = sum(scores)
    if sum_scores <= 0:
        # evitiamo divisione per zero, se fosse tutto <= 0
        return [1.0 / len(players_pool)] * len(players_pool)
    
    classic_probs = [s / sum_scores for s in scores]
    return classic_probs

###############################################################################
# 3) Scelta non bayesiana (euristica + random max 20% del punteggio base)
###############################################################################
def pick_next_player_non_bayesian(players_pool):
    """
    1) calcola base_score = (2×win + xp + form - 0.5×loss)
    2) random extra = random(0, 0.2 * base_score) se base_score>0
    3) score finale = base_score + extra
    """
    if len(players_pool) == 1:
        return players_pool[0]
    
    best_score = float('-inf')
    best_player = None
    
    for p in players_pool:
        base_score = (2.0 * p["partite_vinte"]
                      + p["anni_di_esperienza"]
                      + p["form_index"]
                      - 0.5 * p["partite_perse"])
        
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

###############################################################################
# 4) Funzione: Calcola e mostra la "matrice" di probabilità + doppio grafico a barre
###############################################################################
def do_calcola_prob():
    """
    Calcola (con PyMC) la distribuzione di probabilità su TUTTI i giocatori nel pool (Bayes)
    e la distribuzione classica (score normalizzato).
    Poi le mostra in una matrice (Bayes vs Classico) e in un grafico a barre.
    
    Inoltre, salviamo questi vettori in bayes_history e classic_history
    per poter poi visualizzare l'andamento con i grafici a linee.
    """
    if not pool:
        info_label.config(text="Tutti i giocatori sono stati scelti. Non c'è più nessuno nel pool.")
        return
    
    progress.pack()
    progress.start(10)
    
    def background_calc():
        bayes_probs = build_bayesian_model_and_sample_all(pool)
        classic_probs = compute_classic_probabilities(pool)
        
        # Costruiamo un vettore di lunghezza 10 per ciascuno:
        # Se un giocatore non è nel pool, la prob = 0
        bayes_vector = [0.0]*len(players_data)
        classic_vector = [0.0]*len(players_data)
        
        # Creiamo mappa (giocatore -> indice nel pool)
        #  Così se players_data[i] è in pool[j], salviamo bayes_probs[j]
        for i, p in enumerate(players_data):
            if p in pool:
                j = pool.index(p)
                bayes_vector[i] = bayes_probs[j]
                classic_vector[i] = classic_probs[j]
            else:
                # Non nel pool => 0
                bayes_vector[i] = 0.0
                classic_vector[i] = 0.0
        
        # Salviamo in history
        bayes_history.append(bayes_vector)
        classic_history.append(classic_vector)
        
        def on_done():
            progress.stop()
            progress.pack_forget()
            
            # Matrice di output (nome, bayes, classico) per chi è attualmente nel pool
            txt_lines = []
            for j, pl in enumerate(pool):
                txt_lines.append(
                    f"{pl['name']}: Bayes={bayes_probs[j]:.3f}, Classico={classic_probs[j]:.3f}"
                )
            matrix_output = "\n".join(txt_lines)
            
            info_label.config(text="Distribuzione di probabilità: Bayes vs Classico (prossimo pick).")
            prob_label.config(text=matrix_output)
            
            # Grafico a barre: due serie (Bayes, Classico) affiancate
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

###############################################################################
# 5) Funzione: Esegue la scelta "reale"
###############################################################################
def do_scelta():
    """
    Esegue davvero la scelta del capitano di turno (non bayesiana).
    Rimuove il giocatore scelto dal pool e lo mette nella squadra corrispondente.
    """
    global pool
    
    if not pool:
        info_label.config(text="Non ci sono più giocatori disponibili.")
        calc_prob_button.config(state="disabled")
        choice_button.config(state="disabled")
        return
    
    current_captain, current_team = get_current_captain_and_team()
    picks_done = len(team_first) + len(team_second) - 2
    
    chosen_one = pick_next_player_non_bayesian(pool)
    current_team.append(chosen_one)
    pool.remove(chosen_one)
    
    info_label.config(
        text=f"Pick #{picks_done+1} - Capitano {current_captain['name']} ha scelto: {chosen_one['name']}"
    )
    refresh_player_labels()
    
    # Se finiamo i giocatori ...
    if not pool:
        info_label.config(text="Tutti i giocatori sono stati scelti. Fine selezioni.")
        calc_prob_button.config(state="disabled")
        choice_button.config(state="disabled")

###############################################################################
# 6) Pulsanti "Stampa statistiche bayesiane" e "Stampa statistiche classiche"
###############################################################################
def do_stats_bayes():
    """
    Mostra un grafico a linee per l'andamento delle probabilità bayesiane di TUTTI i giocatori,
    su tutti i 'turni' in cui è stato premuto 'Calcola Probabilità'.
    bayes_history[t][i] => probabilità di i-esimo giocatore al 'turno' t.
    """
    if len(bayes_history) == 0:
        info_label.config(text="Nessun dato bayesiano salvato (non hai mai premuto 'Calcola Probabilità'?).")
        return
    
    # bayes_history è una lista di lunghezza (#chiamate),
    # ogni elemento è un vettore di lunghezza 10 con le probabilità di ogni giocatore
    n_turns = len(bayes_history)
    x_values = range(1, n_turns+1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Andamento Probabilità Bayesiane (per giocatore)")
    
    for i in range(len(players_data)):
        # Costruiamo la serie y = [ bayes_history[t][i] for t in ...]
        y = [bayes_history[t][i] for t in range(n_turns)]
        player_name = players_data[i]["name"]
        plt.plot(x_values, y, label=player_name)
    
    plt.xlabel("Turno # (click Calcola Probabilità)")
    plt.ylabel("Prob. Bayesiana")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def do_stats_classic():
    """
    Mostra un grafico a linee per l'andamento delle probabilità classiche
    di TUTTI i giocatori, su tutti i turni in cui è stato premuto 'Calcola Probabilità'.
    """
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

###############################################################################
# 7) Colleghiamo i pulsanti e avviamo la GUI
###############################################################################
calc_prob_button.config(command=do_calcola_prob)
choice_button.config(command=do_scelta)
stats_bayes_button.config(command=do_stats_bayes)
stats_classic_button.config(command=do_stats_classic)

root.mainloop()
