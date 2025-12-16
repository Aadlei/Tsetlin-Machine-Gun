import os
import subprocess
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# Try importing GTM
try:
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
    HAS_GTM = True
except ImportError:
    print("WARNING: GraphTsetlinMachine not installed. GTM training will fail.")
    HAS_GTM = False

# === CONFIGURATION ===
# Default to 3x3 as requested for verification
DEFAULT_BOARD_DIM = 3
DEFAULT_GAMES = 10000
# User friend's results showed overfitting, so we use a time-based seed
DEFAULT_SEED = 20

def setup_hex(board_dim):
    """Rebuilds the Hex engine for the given dimension."""
    print(f"Building Hex engine for {board_dim}x{board_dim}...")
    subprocess.run(["make", "clean"], cwd="hex", check=True, capture_output=True)
    subprocess.run(["make", f"BOARD_DIM={board_dim}"], cwd="hex", check=True, capture_output=True)

def generate_data(board_dim, n_games, seed, output_file):
    """Generates game data using the C engine."""
    print(f"Generating {n_games} games (Seed: {seed})...")
    cmd = [
        "./scripts/run_hex.sh",
        "--games", str(n_games),
        "--seed", str(seed),
        "--dump-moves", output_file
    ]
    env = os.environ.copy()
    env["BOARD_DIM"] = str(board_dim)
    
    start = time.time()
    subprocess.run(cmd, env=env, check=True)
    print(f"Data generation complete in {time.time() - start:.2f}s")

def load_and_process_data(csv_path, board_dim, offset=0):
    """
    Loads data and reconstructs board states.
    offset: Look at board state 'offset' moves before the end.
    """
    print(f"Processing data with offset={offset}...")
    df = pd.read_csv(csv_path)
    n_nodes = board_dim * board_dim
    
    x_feat = []
    o_feat = []
    labels = []
    
    # Process games
    # Vectorizing this is hard due to variable game lengths, using loop with tqdm
    for game_id, group in tqdm(df.groupby("game_id"), desc="Replaying games"):
        if len(group) <= offset:
            continue
            
        moves = group.iloc[:-offset] if offset > 0 else group
        winner = group["winner"].iloc[0]
        
        p0 = np.zeros(n_nodes, dtype=np.int8)
        p1 = np.zeros(n_nodes, dtype=np.int8)
        
        for _, row in moves.iterrows():
            r, c, p = int(row['row']), int(row['col']), int(row['player'])
            if 0 <= r < board_dim and 0 <= c < board_dim:
                idx = r * board_dim + c
                if p == 0:
                    p0[idx] = 1
                else:
                    p1[idx] = 1
        
        x_feat.append(p0)
        o_feat.append(p1)
        labels.append(winner)
        
    return np.array(x_feat), np.array(o_feat), np.array(labels)

def prepare_graphs(x, o, board_dim, init_with=None):
    if not HAS_GTM: return None
    
    n_samples = len(x)
    n_nodes = board_dim * board_dim
    
    graphs = Graphs(
        n_samples,
        symbols=["Empty", "Player0", "Player1"],
        hypervector_size=1024,
        hypervector_bits=2,
        init_with=init_with
    )
    
    # Hex neighbors
    offsets = [(0, 1), (0, -1), (-1, 1), (1, -1), (-1, 0), (1, 0)]
    adjacency = {}
    
    for r in range(board_dim):
        for c in range(board_dim):
            u = r * board_dim + c
            adjacency[u] = []
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_dim and 0 <= nc < board_dim:
                    adjacency[u].append(nr * board_dim + nc)
    
    for i in range(n_samples):
        graphs.set_number_of_graph_nodes(i, n_nodes)
    graphs.prepare_node_configuration()
    
    for i in range(n_samples):
        for u in range(n_nodes):
            graphs.add_graph_node(i, u, len(adjacency[u]))
    graphs.prepare_edge_configuration()
    
    for i in range(n_samples):
        for u in range(n_nodes):
            for v in adjacency[u]:
                graphs.add_graph_node_edge(i, u, v, 0)
    
    for i in range(n_samples):
        p0_data = x[i]
        p1_data = o[i]
        for u in range(n_nodes):
            if p0_data[u]:
                graphs.add_graph_node_property(i, u, "Player0")
            elif p1_data[u]:
                graphs.add_graph_node_property(i, u, "Player1")
            else:
                graphs.add_graph_node_property(i, u, "Empty")
                
    graphs.encode()
    return graphs

def run_experiment(args):
    # Setup
    setup_hex(args.dim)
    output_file = f"runs/hex_{args.dim}_{args.games}.csv"
    
    if not os.path.exists("runs"):
        os.makedirs("runs")
        
    generate_data(args.dim, args.games, args.seed, output_file)
    
    results = []
    
    # Offsets to evaluate
    offsets = [0, 2, 5]
    
    for offset in offsets:
        print(f"\n{'='*30}\nEvaluating Offset: {offset}\n{'='*30}")
        
        # 1. Prepare Data
        X_p0, X_p1, Y = load_and_process_data(output_file, args.dim, offset)
        
        # Split (50/50)
        split_idx = len(Y) // 2
        X_train = (X_p0[:split_idx], X_p1[:split_idx])
        Y_train = Y[:split_idx]
        X_test = (X_p0[split_idx:], X_p1[split_idx:])
        Y_test = Y[split_idx:]
        
        print(f"Train samples: {len(Y_train)}, Test samples: {len(Y_test)}")
        
        if HAS_GTM:
            # 2. Graphs
            g_train = prepare_graphs(X_train[0], X_train[1], args.dim)
            g_test = prepare_graphs(X_test[0], X_test[1], args.dim, init_with=g_train)
            
            # 3. Train
            tm = MultiClassGraphTsetlinMachine(
                number_of_clauses=args.clauses,
                T=args.T,
                s=args.s,
                depth=args.depth,
                message_size=512,
                message_bits=2,
                max_included_literals=32,
                grid=(16*13, 1, 1),
                block=(128, 1, 1)
            )
            
            history = []
            for ep in range(args.epochs):
                tm.fit(g_train, Y_train, epochs=1, incremental=True)
                # Check train/test acc randomly or every epoch?
                # Friend plotted every epoch.
                pred_train = tm.predict(g_train)
                acc_train = accuracy_score(Y_train, pred_train)
                history.append(acc_train * 100)
                if (ep+1) % 5 == 0:
                     print(f"Epoch {ep+1}: {acc_train*100:.2f}%")
                     
            # 4. Test
            preds = tm.predict(g_test)
            acc = accuracy_score(Y_test, preds)
            cm = confusion_matrix(Y_test, preds)
            
            print(f"FINAL Test Accuracy: {acc*100:.2f}%")
            
            results.append({
                "offset": offset,
                "accuracy": acc * 100,
                "history": history,
                "confusion": cm
            })
            
    # Visualizations
    if results:
        # Learning Curves
        plt.figure(figsize=(10, 6))
        for res in results:
            plt.plot(res['history'], label=f"Offset {res['offset']} (Final: {res['accuracy']:.1f}%)")
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy %")
        plt.title(f"Learning Curves (Dim {args.dim}x{args.dim})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("learning_curves.png")
        print("Saved learning_curves.png")
        
        # Confusion Matrices
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
        if len(results) == 1: axes = [axes]
        for i, res in enumerate(results):
            sns.heatmap(res['confusion'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f"Offset {res['offset']}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        plt.savefig("confusion_matrices.png")
        print("Saved confusion_matrices.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=DEFAULT_BOARD_DIM, help="Board dimension")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="Number of games")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    
    # GTM Hyperparams (Default for 3x3)
    parser.add_argument("--clauses", type=int, default=400)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--s", type=float, default=5.0)
    parser.add_argument("--depth", type=int, default=5)
    
    args = parser.parse_args()
    
    print("=== SCALING ADVICE ===")
    print("For 3x3: Clauses=400, T=2000, Depth=5")
    print("For 11x11: Clauses=5000+, T=4000+, Depth=12+")
    print("======================\n")
    
    run_experiment(args)
