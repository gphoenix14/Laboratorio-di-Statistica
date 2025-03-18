import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking
from tensorflow.keras.optimizers import Adam

# Imposta i seed per la riproducibilità
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Dati dei giocatori
players_data = [
    {"name": "Alex",     "anni_di_esperienza": 2,  "partite_vinte": 6,   "partite_perse": 4,  "form_index": 6},
    {"name": "Bruno",    "anni_di_esperienza": 3,  "partite_vinte": 9,   "partite_perse": 3,  "form_index": 7},
    {"name": "Carlo",    "anni_di_esperienza": 6,  "partite_vinte": 14,  "partite_perse": 6,  "form_index": 9},
    {"name": "Dario",    "anni_di_esperienza": 1,  "partite_vinte": 2,   "partite_perse": 7,  "form_index": 4},
    {"name": "Enea",     "anni_di_esperienza": 5,  "partite_vinte": 11,  "partite_perse": 4,  "form_index": 8},
    {"name": "Fabio",    "anni_di_esperienza": 7,  "partite_vinte": 16,  "partite_perse": 10, "form_index": 10},
    {"name": "Gino",     "anni_di_esperienza": 3,  "partite_vinte": 5,   "partite_perse": 8,  "form_index": 5},
    {"name": "Marco",    "anni_di_esperienza": 9,  "partite_vinte": 21,  "partite_perse": 11, "form_index": 9},
    {"name": "Paolo",    "anni_di_esperienza": 6,  "partite_vinte": 13,  "partite_perse": 7,  "form_index": 8},
    {"name": "Rocco",    "anni_di_esperienza": 2,  "partite_vinte": 3,   "partite_perse": 5,  "form_index": 6},
    {"name": "Federica", "anni_di_esperienza": 3,  "partite_vinte": 7,   "partite_perse": 4,  "form_index": 7},
    {"name": "Rosita",   "anni_di_esperienza": 6,  "partite_vinte": 10,  "partite_perse": 5,  "form_index": 5},
    {"name": "Michela",  "anni_di_esperienza": 7,  "partite_vinte": 12,  "partite_perse": 7,  "form_index": 9},
    {"name": "Roberta",  "anni_di_esperienza": 4,  "partite_vinte": 6,   "partite_perse": 5,  "form_index": 6},
    {"name": "Anna",     "anni_di_esperienza": 5,  "partite_vinte": 10,  "partite_perse": 4,  "form_index": 7},
    {"name": "Lorena",   "anni_di_esperienza": 8,  "partite_vinte": 18,  "partite_perse": 10, "form_index": 8},
    {"name": "Marlena",  "anni_di_esperienza": 3,  "partite_vinte": 8,   "partite_perse": 6,  "form_index": 5},
    {"name": "Sara",     "anni_di_esperienza": 5,  "partite_vinte": 15,  "partite_perse": 8,  "form_index": 6},
    {"name": "Maria",    "anni_di_esperienza": 6,  "partite_vinte": 14,  "partite_perse": 7,  "form_index": 7},
    {"name": "Milena",   "anni_di_esperienza": 4,  "partite_vinte": 9,   "partite_perse": 6,  "form_index": 7},
]

# Funzione per estrarre le feature del giocatore
def get_features(player):
    return np.array([
        player["anni_di_esperienza"],
        player["partite_vinte"],
        player["partite_perse"],
        player["form_index"]
    ], dtype=np.float32)

# Funzione di scelta "classica"
K = 7.0
def classical_pick(pool, pref):
    best_score = -1e9
    best = None
    for p in pool:
        base = 2.0 * p["partite_vinte"] + p["anni_di_esperienza"] + p["form_index"] - 0.5 * p["partite_perse"]
        if pref == "vittorie":
            base += (K - 1.0) * (2.0 * p["partite_vinte"])
        elif pref == "esperienza":
            base += (K - 1.0) * (p["anni_di_esperienza"])
        elif pref == "forma":
            base += (K - 1.0) * (p["form_index"])
        elif pref == "perdite":
            base += (K - 1.0) * (-0.5 * p["partite_perse"])
        extra = random.uniform(0, 0.2 * base) if base > 0 else 0.0
        score = base + extra
        if score > best_score:
            best_score = score
            best = p
    return best

# Parametro per la lunghezza massima del contesto
max_turns = 10

# Funzione per effettuare il padding del contesto in modo che sia sempre (max_turns, 4)
def pad_context(context):
    if len(context) == 0:
        return np.zeros((max_turns, 4), dtype=np.float32)
    context_arr = np.array(context, dtype=np.float32)
    if context_arr.ndim == 1:
        context_arr = context_arr.reshape((-1, 4))
    if context_arr.shape[0] >= max_turns:
        return context_arr[-max_turns:]
    else:
        pad = np.zeros((max_turns - context_arr.shape[0], 4), dtype=np.float32)
        return np.concatenate([pad, context_arr], axis=0)

# Simula una sessione per generare dati di training
def simulate_episode():
    samples_context = []
    samples_candidate = []
    samples_label = []
    pool = players_data.copy()
    random.shuffle(pool)
    pref = random.choice(["esperienza", "vittorie", "forma", "perdite"])
    context = []  # sequenza delle feature dei giocatori già scelti
    while pool:
        chosen = classical_pick(pool, pref)
        for p in pool:
            context_pad = pad_context(context)
            samples_context.append(context_pad)
            samples_candidate.append(get_features(p))
            label = 1.0 if p == chosen else 0.0
            samples_label.append(label)
        context.append(get_features(chosen))
        pool.remove(chosen)
    return samples_context, samples_candidate, samples_label

# Genera dati di training simulando più episodi
num_episodes = 100
X_context_list, X_candidate_list, y_list = [], [], []
for _ in range(num_episodes):
    ctx, cand, lbl = simulate_episode()
    X_context_list.extend(ctx)
    X_candidate_list.extend(cand)
    y_list.extend(lbl)

X_context = np.array(X_context_list)   # Shape: (num_samples, max_turns, 4)
X_candidate = np.array(X_candidate_list)  # Shape: (num_samples, 4)
y = np.array(y_list)                   # Shape: (num_samples,)

# Costruzione del modello: una LSTM per il contesto e Dense per il candidato
context_input = Input(shape=(max_turns, 4))
mask = Masking(mask_value=0.0)(context_input)
lstm_out = LSTM(16)(mask)
candidate_input = Input(shape=(4,))
concat = Concatenate()([lstm_out, candidate_input])
x = Dense(16, activation='relu')(concat)
x = Dense(8, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[context_input, candidate_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit([X_context, X_candidate], y, epochs=5, batch_size=64, verbose=1)

# Funzione per predire le probabilità dei candidati dato un contesto
def predict_candidates(context, candidate_pool):
    context_pad = pad_context(context)
    scores = []
    for p in candidate_pool:
        features = get_features(p)
        pred = model.predict([np.expand_dims(context_pad, axis=0),
                              np.expand_dims(features, axis=0)],
                             verbose=0)
        scores.append(pred[0][0])
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    return probs

# Esempio di utilizzo: contesto vuoto e pool casuale
pool_test = players_data.copy()
random.shuffle(pool_test)
context_test = []  # nessun giocatore scelto
probs = predict_candidates(context_test, pool_test)
for p, prob in zip(pool_test, probs):
    print(f"{p['name']}: {prob:.3f}")
