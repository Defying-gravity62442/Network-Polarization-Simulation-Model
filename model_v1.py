import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class PolarizationModel:
    def __init__(
        self,
        N: int,
        alpha: Tuple[float, float, float] = (0.6, 0.1, 0.3),
        theta: Tuple[float, float, float, float, float] = (-1.0, 3.0, 2.0, 4.0, 0.5),
        *,
        delta0: float = 0.1,    # Lowered base tolerance for pruning (harder to keep edges)
        gamma_tol: float = 0.5,  # Increased stubbornness factor to tolerate less discord
        gamma_taper: float = 1.0,
        mu_s: float = 0.3,
        sigma_s: float = 0.15,
        EI0: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if N < 3:
            raise ValueError("Need at least 3 nodes so zealots are distinct from normal users.")
        self.N = N
        self.Z_PLUS = 0
        self.Z_MINUS = N - 1
        self.rng = rng or np.random.default_rng()

        k = 4
        p = 0.1
        self.G = nx.watts_strogatz_graph(N, k, p, seed=self.rng)

        self.w = self.rng.uniform(low=0.0, high=1.0, size=N)
        self.w[self.Z_PLUS] = 1.0
        self.w[self.Z_MINUS] = 0.0

        self.s = np.clip(self.rng.normal(mu_s, sigma_s, size=N), 0.0, 1.0)
        self.s[self.Z_PLUS] = 1.0
        self.s[self.Z_MINUS] = 1.0

        self.alpha1, self.alpha2, self.alpha3 = alpha
        self.theta0, self.theta1, self.theta2, self.theta3, self.theta4 = theta

        self.delta0 = delta0
        self.gamma_tol = gamma_tol
        self.gamma_taper = gamma_taper

        self.EI = EI0
        self.rho_EI = 0.7
        self.sigma_EI = 1.0

        self.t = 0

    def _aa_weight(self, deg: int) -> float:
        return 1.0 if deg <= 1 else 1.0 / np.log(deg)

    def _normalized_adamic_adar(self, i: int, j: int) -> float:
        Ni = set(self.G.neighbors(i))
        Nj = set(self.G.neighbors(j))
        common = Ni & Nj
        if not common:
            return 0.0
        union = Ni | Nj
        num = sum(self._aa_weight(self.G.degree[z]) for z in common)
        den = sum(self._aa_weight(self.G.degree[z]) for z in union)
        return num / den if den else 0.0

    def _candidate_set(self, i: int) -> List[int]:
        dist2 = set()
        for nbr in self.G.neighbors(i):
            dist2.update(self.G.neighbors(nbr))
        dist2.discard(i)
        dist2.difference_update(self.G.neighbors(i))
        return list(dist2)

    def _recommendations(self) -> Dict[int, int]:
        rec = {}
        for i in range(self.N):
            if i in (self.Z_PLUS, self.Z_MINUS):
                continue
            candidates = self._candidate_set(i)
            if not candidates:
                continue
            scores = []
            for j in candidates:
                aa = self._normalized_adamic_adar(i, j)
                pop = self.G.degree[j] / max(1, self.N - 1)
                hom = 1.0 - abs(self.w[i] - self.w[j])
                score = self.alpha1 * aa + self.alpha2 * pop + self.alpha3 * hom
                scores.append(score)
            j_star = candidates[int(np.argmax(scores))]
            rec[i] = j_star
        return rec

    def _acceptance_probability(self, i: int, j: int, cn_ij: int) -> float:
        s_i = self.s[i]
        deg_i = max(0, self.G.degree[i])
        if deg_i:
            avg_friend_op = np.mean([self.w[k] for k in self.G.neighbors(i)])
            rho_i = 1.0 - abs(self.w[i] - avg_friend_op)
        else:
            avg_friend_op = 0.5
            rho_i = 0.0

        if cn_ij:
            mutuals = set(self.G.neighbors(i)) & set(self.G.neighbors(j))
            est_w_j = np.mean([self.w[k] for k in mutuals])
        else:
            est_w_j = avg_friend_op
        d_hat = abs(self.w[i] - est_w_j)

        x = (
            self.theta0
            - self.theta1 * s_i
            + self.theta2 * rho_i * cn_ij / (deg_i + 1)
            + self.theta3 * (1.0 - d_hat)
            + self.theta4 * self.EI
        )
        return 1.0 / (1.0 + np.exp(-x))

    def _add_edges_from_recommendations(self, rec: Dict[int, int]) -> int:
        """Add recommended edges if accepted, and return count of new edges added."""
        count_new_edges = 0
        for i, j in rec.items():
            if self.G.has_edge(i, j):
                continue  # already friends
            cn_ij = len(set(self.G.neighbors(i)) & set(self.G.neighbors(j)))
            p_acc = self._acceptance_probability(i, j, cn_ij)
            if self.rng.random() < p_acc:
                self.G.add_edge(i, j)
                count_new_edges += 1
        return count_new_edges

    def _update_opinions(self):
        new_w = self.w.copy()
        for i in range(self.N):
            if i in (self.Z_PLUS, self.Z_MINUS):
                continue
            deg_i = self.G.degree[i]
            if deg_i == 0:
                continue
            avg_neighbor = np.mean([self.w[j] for j in self.G.neighbors(i)])
            taper = 1.0 - abs(2 * self.w[i] - 1) ** self.gamma_taper
            delta = (1.0 - self.s[i]) * taper * (avg_neighbor - self.w[i])
            new_w[i] += delta
        self.w = np.clip(new_w, 0.0, 1.0)

    def _tolerance(self, i: int) -> float:
        return np.clip(self.delta0 + self.gamma_tol * self.s[i], 0.0, 1.0)

    def _prune_edges(self, passes=3):
        for pass_num in range(passes):
            to_remove = set()
            for i, j in list(self.G.edges()):
                d_ij = abs(self.w[i] - self.w[j])
                delta_i = self._tolerance(i)
                delta_j = self._tolerance(j)

                def p_cut(delta_u):
                    if d_ij <= delta_u:
                        return 0.0
                    #diff = (d_ij - delta_u) / (1.0 - delta_u)
                    #return diff ** 3  # steeper function
                    return 1.0
                p_i = p_cut(delta_i)
                p_j = p_cut(delta_j)

                # Edge pruned if either node decides to prune
                if self.rng.random() < p_i or self.rng.random() < p_j:
                    to_remove.add((i, j))

            if to_remove:
                print(f"Step {self.t} pass {pass_num}: Pruning {len(to_remove)} edges")
                self.G.remove_edges_from(to_remove)
            else:
                break

    def step(self):
        # 1. recommendation
        rec = self._recommendations()

        # 2. acceptance decision
        num_added = self._add_edges_from_recommendations(rec)
        print(f"Step {self.t}: Added {num_added} new edges")

        # 3. opinion update
        self._update_opinions()

        # 4. prune edges (use improved pruning method)
        self._prune_edges(passes=3)

        # 5. evolve external influence EI_t (simple AR(1))
        self.EI = self.rho_EI * self.EI + self.sigma_EI * self.rng.normal()

        self.t += 1

    def spectral_gap(self) -> float:
        if self.G.number_of_edges() == 0:
            return 0.0
        L = nx.normalized_laplacian_matrix(self.G).toarray()
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()
        return float(eigvals[1])

    def average_opinion(self) -> float:
        return float(self.w.mean())

    def run(self, T: int):
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
        pos = nx.spring_layout(self.G, seed=42)

        for t in range(T):
            self.step()
            ax.clear()
            nodes = self.G.nodes()
            colors = [self.w[n] for n in nodes]
            nx.draw(
                self.G, pos,
                node_color=colors, cmap=plt.cm.coolwarm,
                with_labels=False, node_size=50, edge_color='gray', alpha=0.7,
                ax=ax
            )
            ax.set_title(f"Step {t} — Spectral gap: {self.spectral_gap():.3f} — Avg opinion: {self.average_opinion():.3f}")
            plt.pause(0.1)

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    model = PolarizationModel(N=100, rng=np.random.default_rng(42))
    model.run(T=100)
