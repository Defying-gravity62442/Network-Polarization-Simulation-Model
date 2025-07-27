import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# ----------------------------
# Parameter Initialization
# ----------------------------
def initialize_params():
    return {
        'N0': 16,        # initial founders
        'p0': 0.3,       # intra-cluster ER prob
        'eps': 0.1,      # interaction threshold
        'mu_max': 0.05,  # max update rate
        'alpha': 20,     # deletion steepness
        'p_add': 0.5,   # rewiring prob per node
        'max_users': 100,
        'T_evo': 200,
        'seed': 42
    }

# ----------------------------
# Simulation: arrival + evolution + metrics
# ----------------------------
def simulate(params):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    N0, p0, eps = params['N0'], params['p0'], params['eps']
    mu_max, alpha, p_add = params['mu_max'], params['alpha'], params['p_add']
    max_u, T = params['max_users'], params['T_evo']

    # Initialize founders in two opinion clusters
    half = N0 // 2
    w = np.concatenate([
        np.clip(np.random.normal(0.1, 0.02, half), 0,1),
        np.clip(np.random.normal(0.9, 0.02, N0-half), 0,1)
    ])
    G = nx.Graph()
    G.add_nodes_from(range(N0))
    # Connect founders by ER
    for i in range(N0):
        for j in range(i):
            p = p0 if ((i<half and j<half) or (i>=half and j>=half)) else p0*0.2
            if random.random() < p:
                G.add_edge(i, j)

    trace = []
    metrics = {
        'overall_var': [],
        'within_var': [],
        'between_dist': []
    }
    # Evolution steps
    for t in range(T):
        # Arrival
        if G.number_of_nodes() < max_u:
            i_new = G.number_of_nodes()
            new_op = random.random()
            w = np.append(w, new_op)
            G.add_node(i_new)
            # connect to 5 nearest in opinion
            dists = np.abs(w[:i_new] - new_op)
            for j in np.argsort(dists)[:5]:
                G.add_edge(i_new, j)
        # Probabilistic addition
        for i in range(len(w)):
            if random.random() < p_add:
                sim = np.exp(-((w - w[i])**2)/(2*eps**2))
                sim[i] = 0
                probs = sim/sim.sum()
                j = np.random.choice(len(w), p=probs)
                if not G.has_edge(i, j):
                    G.add_edge(i, j)
        # Opinion update
        old = w.copy()
        for i, j in G.edges():
            diff = old[j] - old[i]
            if abs(diff) <= eps:
                ext_i = 2 * abs(old[i] - 0.5)
                ext_j = 2 * abs(old[j] - 0.5)
                ext_i = min(ext_i, 1.0)
                ext_j = min(ext_j, 1.0)

                mu_i = mu_max * (1.0 - ext_i)
                mu_j = mu_max * (1.0 - ext_j)

                w[i] += mu_i * diff
                w[j] -= mu_j * diff
        # Probabilistic deletion
        for i, j in list(G.edges()):
            dw = abs(w[i]-w[j])
            p_del = 1/(1+np.exp(-alpha*(dw-eps)))
            if random.random() < p_del:
                G.remove_edge(i, j)
        
        # Record trace
        trace.append((G.copy(), w.copy()))
        # Compute metrics
        overall_var = np.var(w)
        # clusters by threshold 0.5
        c0 = w[w<=0.5]; c1 = w[w>0.5]
        var0 = np.var(c0) if len(c0)>1 else 0
        var1 = np.var(c1) if len(c1)>1 else 0
        within_var = (var0 + var1)/2
        between_dist = abs(c0.mean() - c1.mean()) if len(c0)>0 and len(c1)>0 else 0
        metrics['overall_var'].append(overall_var)
        metrics['within_var'].append(within_var)
        metrics['between_dist'].append(between_dist)
    return trace, metrics

# ----------------------------
# Animate opinion and network
# ----------------------------
def animate(trace, pos):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))
    Nmax = max(len(w) for _, w in trace)
    y_pos = np.arange(Nmax)
    for step, (G_t, w_t) in enumerate(trace, start=1):
        ax.clear()
        # draw edges
        for i,j in G_t.edges():
            ax.plot([w_t[i],w_t[j]],[y_pos[i],y_pos[j]],color='gray',alpha=0.3)
        # draw nodes
        ax.scatter(w_t, y_pos[:len(w_t)], c=w_t, cmap='bwr',vmin=0,vmax=1, s=80)
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-1,Nmax)
        ax.set_xlabel('Opinion'); ax.set_ylabel('Node')
        ax.set_title(f'Step {step}/{len(trace)}')
        plt.pause(0.1)
    plt.ioff(); plt.show()

# ----------------------------
# Plot metrics over time
# ----------------------------
def plot_metrics(metrics):
    t = range(len(metrics['overall_var']))
    plt.figure(figsize=(8,6))
    plt.plot(t, metrics['overall_var'], label='Overall Var')
    plt.plot(t, metrics['within_var'], label='Within-Cluster Var')
    plt.plot(t, metrics['between_dist'], label='Between-Cluster Dist')
    plt.xlabel('Time Step'); plt.ylabel('Value')
    plt.legend(); plt.title('Polarization Metrics Over Time')
    plt.show()

# ----------------------------
# Main execution
# ----------------------------
if __name__ == '__main__':
    params = initialize_params()
    trace, metrics = simulate(params)
    final_G, _ = trace[-1]
    pos = nx.spring_layout(final_G, seed=params['seed'])
    animate(trace, pos)
    plot_metrics(metrics)
    # compute final metrics
    compute_metrics = lambda m: None  # stub
    print('Simulation complete.')

# ----------------------------
# Parameter Reference
# ----------------------------
# N0       | 10–30        
# p0       | 0.1–0.5      
# eps      | 0.05–0.2     
# mu_max   | 0.01–0.1     
# alpha    | 10–30        
# p_add    | 0.01–0.1     
# max_users| 50–200       
# T_evo    | 100–500      