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

###############################################################################
# 2) Scelta casuale dei capitani e del soggetto di test
###############################################################################
import random
random.shuffle(players_data)

two_captains = random.sample(players_data, 2)
captain1, captain2 = two_captains[0], two_captains[1]

sorted_names = sorted([captain1["name"], captain2["name"]])
if sorted_names[0] == captain1["name"]:
    first_captain, second_captain = captain1, captain2
else:
    first_captain, second_captain = captain2, captain1

remaining_players = [p for p in players_data if p not in [captain1, captain2]]

test_subject = random.choice(remaining_players)

team_first = [first_captain]
team_second = [second_captain]

###############################################################################
# 3) Variabili globali per la GUI
###############################################################################
root = tk.Tk()
root.title("Selezione Squadre - Calcola Probabilità & Scelta")
root.geometry("1100x700")

BIG_FONT = ("Arial", 14)
NORMAL_FONT = ("Arial", 12)
SMALL_FONT = ("Arial", 10)

p_calcolata = 0.0            # Ultima probabilità calcolata
probabilities_history = []    # Storico delle probabilità calcolate

frame_top = tk.Frame(root)
frame_top.pack(pady=10)

info_label = tk.Label(frame_top, text="Premi i pulsanti per iniziare", font=BIG_FONT)
info_label.pack()

prob_label = tk.Label(frame_top, text="", font=NORMAL_FONT, fg="blue")
prob_label.pack(pady=5)

# Due pulsanti: "Calcola Probabilità" e "Scelta"
calc_prob_button = tk.Button(frame_top, text="Calcola Probabilità", font=NORMAL_FONT)
calc_prob_button.pack(side="left", padx=20)

choice_button = tk.Button(frame_top, text="Scelta", font=NORMAL_FONT)
choice_button.pack(side="left", padx=20)

# Barra di caricamento
progress = ttk.Progressbar(frame_top, orient='horizontal', length=300, mode='indeterminate')
progress.pack(pady=5)
progress.pack_forget()

frame_players = tk.Frame(root)
frame_players.pack(pady=10)

player_labels = []

###############################################################################
# 4) Creazione dei label per i giocatori
###############################################################################
def create_player_labels():
    for i, player in enumerate(players_data):
        lbl = tk.Label(frame_players, text="", font=SMALL_FONT, width=45, borderwidth=2, relief="groove")
        lbl.grid(row=i // 2, column=i % 2, padx=5, pady=5)
        player_labels.append(lbl)

create_player_labels()

###############################################################################
# 5) Funzioni di supporto
###############################################################################
def refresh_player_labels():
    """Aggiorna il testo e i colori dei label in base alle scelte già fatte."""
    for lbl, player in zip(player_labels, players_data):
        txt = (f"{player['name']} "
               f"(Exp={player['anni_di_esperienza']}, Win={player['partite_vinte']}, "
               f"Lost={player['partite_perse']}, Form={player['form_index']})")
        lbl.config(text=txt, fg="black")
    
    # Capitani
    for lbl, player in zip(player_labels, players_data):
        if player == first_captain:
            lbl.config(fg="red")
        elif player == second_captain:
            lbl.config(fg="green")
    
    # Soggetto di test
    for lbl, player in zip(player_labels, players_data):
        if player == test_subject:
            lbl.config(fg="blue")
    
    # Giocatori scelti (tranne i capitani)
    chosen_players = team_first + team_second
    for lbl, player in zip(player_labels, players_data):
        if (player in chosen_players) and (player not in [first_captain, second_captain]):
            lbl.config(fg="gray")

def get_current_captain_and_team():
    """Stabilisce quale capitano deve scegliere in base al numero di pick già fatti."""
    picks_done = len(team_first) + len(team_second) - 2  # -2 per i due capitani
    if picks_done % 2 == 0:
        return first_captain, team_first
    else:
        return second_captain, team_second

###############################################################################
# 6) Modello Bayesiano: calcolo probabilità
###############################################################################
def build_bayesian_model_and_sample(players_pool, test_player, picks_done):
    """Calcola la probabilità che test_player venga scelto NEL PROSSIMO PICK."""
    if test_player not in players_pool:
        return 0.0
    
    xp = np.array([p["anni_di_esperienza"] for p in players_pool])
    wins = np.array([p["partite_vinte"] for p in players_pool])
    losses = np.array([p["partite_perse"] for p in players_pool])
    form = np.array([p["form_index"] for p in players_pool])
    
    test_index = players_pool.index(test_player)
    
    # Creiamo un one-hot "statico" come array NumPy
    one_hot_np = np.zeros(len(players_pool), dtype=float)
    one_hot_np[test_index] = 1.0
    
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        
        beta_exp = pm.Normal("beta_exp", mu=1, sigma=1)
        beta_win = pm.Normal("beta_win", mu=2, sigma=1)
        beta_loss = pm.Normal("beta_loss", mu=-1, sigma=1)
        beta_form = pm.Normal("beta_form", mu=1, sigma=1)
        
        # Bonus che cresce con picks_done SOLO per test_subject
        beta_pool = pm.Normal("beta_pool", mu=1, sigma=1)
        
        base_score = (alpha
                      + beta_exp*xp
                      + beta_win*wins
                      + beta_loss*losses
                      + beta_form*form)
        
        # Creiamo una costante PyMC dalla one_hot e la sommiamo
        oh_const = pm.math.constant(one_hot_np)  # shape: (len(players_pool),)
        # final_score è base_score + oh_const * beta_pool * picks_done
        final_score = base_score + oh_const * beta_pool * picks_done
        
        probs = pm.Deterministic("probs", pm.math.softmax(final_score))
        
        # Facciamo un sampling MCMC semplice
        trace = pm.sample(draws=400, tune=200, chains=1, progressbar=False, random_seed=42)
    
    posterior_probs = trace.posterior["probs"].values[0]  # [n_draws, pool_size]
    p_test = posterior_probs[:, test_index].mean()
    return p_test

###############################################################################
# 7) Scelta (non bayesiana) del giocatore
###############################################################################
def pick_next_player_non_bayesian(players_pool):
    """Algoritmo semplicissimo che favorisce vittorie, esperienza, form, e penalizza un po' le perse."""
    if len(players_pool) == 1:
        return players_pool[0]
    
    best_player = None
    best_score = float('-inf')
    
    for p in players_pool:
        import random
        score = (2.0*p["partite_vinte"]
                 + p["anni_di_esperienza"]
                 + p["form_index"]
                 - 0.5*p["partite_perse"]
                 + random.uniform(0,5))  # piccola casualità
        if score > best_score:
            best_score = score
            best_player = p
    
    return best_player

###############################################################################
# 8) LOGICA GUI: due pulsanti separati (Calcola Probabilità, Scelta)
###############################################################################
pool = remaining_players[:]
refresh_player_labels()

def do_calcola_prob():
    """
    Quando clicco su 'Calcola Probabilità', calcolo e mostro la prob. che test_subject
    venga scelto NEL PROSSIMO PICK dal capitano di turno (senza effettuare la scelta).
    """
    if test_subject not in pool:
        info_label.config(text=f"{test_subject['name']} è già stato scelto. Non serve calcolare ancora.")
        return
    
    # Mostriamo la barra di caricamento mentre calcoliamo
    progress.pack()
    progress.start(10)
    
    # Calcolo in un thread separato
    def background_probability():
        current_captain, _ = get_current_captain_and_team()
        picks_done = len(team_first) + len(team_second) - 2
        
        p_test_local = build_bayesian_model_and_sample(pool, test_subject, picks_done)
        
        def on_done():
            progress.stop()
            progress.pack_forget()
            
            info_label.config(
                text=f"Probabilità che {test_subject['name']} venga scelto nel prossimo pick (capitano {current_captain['name']}):"
            )
            prob_label.config(text=f"{p_test_local:.3f}")
            
        root.after(0, on_done)
    
    threading.Thread(target=background_probability).start()

def do_scelta():
    """
    Quando clicco su 'Scelta', faccio effettivamente scegliere un giocatore
    (non bayesiano) dal capitano di turno.
    """
    global pool
    
    if test_subject not in pool:
        info_label.config(text=f"{test_subject['name']} è già stato scelto. Fine selezioni.")
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
    
    # Se è il test_subject, chiudiamo
    if chosen_one == test_subject:
        info_label.config(text=f"{test_subject['name']} è stato scelto! Fine selezioni.")
        calc_prob_button.config(state="disabled")
        choice_button.config(state="disabled")
        
        # Mostriamo il grafico finale delle probabilità calcolate
        # (prendiamo le info dal prob_label se vogliamo storicizzare).
        # Ma finora non abbiamo salvato la storia. Se vogliamo farlo,
        # potremmo salvare un log a ogni "Calcola Probabilità".
        # Per semplicità, mostriamo un grafico con una sola linea se abbiamo
        # accumulato le probabilità in probabilities_history. Facciamolo!
        
        if probabilities_history:
            plt.figure()
            plt.title(f"Andamento Probabilità di {test_subject['name']}")
            plt.plot(range(1, len(probabilities_history)+1), probabilities_history, marker='o')
            plt.xlabel("Calcolo #")
            plt.ylabel("Probabilità Test Subject")
            plt.grid(True)
            plt.show()

###############################################################################
# 9) Salvare lo storico delle probabilità (opzionale)
###############################################################################
# Se vogliamo "accumulare" le probabilità calcolate, possiamo farlo
# in do_calcola_prob(). Aggiorniamo do_calcola_prob() in modo
# da salvare p_test_local nella lista probabilities_history.

def do_calcola_prob():
    if test_subject not in pool:
        info_label.config(text=f"{test_subject['name']} è già stato scelto. Non serve calcolare ancora.")
        return
    
    progress.pack()
    progress.start(10)
    
    def background_probability():
        current_captain, _ = get_current_captain_and_team()
        picks_done = len(team_first) + len(team_second) - 2
        
        p_test_local = build_bayesian_model_and_sample(pool, test_subject, picks_done)
        
        # Salviamo in probabilities_history
        probabilities_history.append(p_test_local)
        
        def on_done():
            progress.stop()
            progress.pack_forget()
            
            info_label.config(
                text=f"Probabilità che {test_subject['name']} venga scelto nel prossimo pick (capitano {current_captain['name']}):"
            )
            prob_label.config(text=f"{p_test_local:.3f}")
            
        root.after(0, on_done)
    
    threading.Thread(target=background_probability).start()

###############################################################################
# 10) Colleghiamo i pulsanti e avviamo la GUI
###############################################################################
calc_prob_button.config(command=do_calcola_prob)
choice_button.config(command=do_scelta)

refresh_player_labels()

root.mainloop()
