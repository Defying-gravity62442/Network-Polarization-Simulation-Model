import numpy as np
import networkx as nx
from typing import Tuple, List, Dict


class PolarizationModel:
    """Opinion–network co‑evolution with zealots, stubbornness and recommendation.

    Parameters
    ----------
    N : int
        Number of vertices.  Node 0 is zealot +1, node N-1 zealot 0.
    alpha : Tuple[float, float, float]
        Weights (α₁, α₂, α₃) for (normalised Adamic–Adar, popularity, homophily)
        in the recommendation score.
    theta : Tuple[float, float, float, float, float]
        Logistic‑acceptance coefficients (θ₀…θ₄).
    delta0 : float
        Base tolerance for unfollow (0 ≤ δ₀ < 1).
    gamma_tol : float
        Add‑on tolerance factor per unit stubbornness (γ in the note).
    gamma_taper : float
        Exponent γ in the tapering factor (1 ≤ γ ≤ 2 typically).
    mu_s, sigma_s : float
        Mean and std‑dev of the initial stubbornness distribution (clipped to [0,1]).
    EI0 : float
        Initial external‑influence value.
    rng : np.random.Generator, optional
        Random number generator (for reproducibility).
    """

    def __init__(
        self,
        N: int,
        alpha: Tuple[float, float, float] = (0.6, 0.1, 0.3),
        theta: Tuple[float, float, float, float, float] = (0.0, 2.0, 2.0, 2.0, 0.5),
        *,
        delta0: float = 0.4,
        gamma_tol: float = 0.3,
        gamma_taper: float = 1.0,
        mu_s: float = 0.3,
        sigma_s: float = 0.15,
        EI0: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if N < 3:
            raise ValueError("Need at least 3 nodes so zealots are distinct from normal users.")
        self.N = N
        self.Z_PLUS = 0         # zealot with opinion 1
        self.Z_MINUS = N - 1    # zealot with opinion 0

        self.G = nx.Graph()
        self.G.add_nodes_from(range(N))

        self.rng = rng or np.random.default_rng()

        # Opinions / beliefs w in [0,1]
        self.w = self.rng.uniform(low=0.0, high=1.0, size=N)
        self.w[self.Z_PLUS] = 1.0
        self.w[self.Z_MINUS] = 0.0

        # Stubbornness s in [0,1]
        self.s = np.clip(self.rng.normal(mu_s, sigma_s, size=N), 0.0, 1.0)
        self.s[self.Z_PLUS] = 1.0  # zealots unchangeable
        self.s[self.Z_MINUS] = 1.0

        self.alpha1, self.alpha2, self.alpha3 = alpha
        self.theta0, self.theta1, self.theta2, self.theta3, self.theta4 = theta

        self.delta0 = delta0
        self.gamma_tol = gamma_tol
        self.gamma_taper = gamma_taper

        # External‑influence AR(1): EI_{t+1} = ρ EI_t + ε
        self.EI = EI0
        self.rho_EI = 0.7
        self.sigma_EI = 1.0

        self.t = 0  # time step counter

    # ---------------------------------------------------------------------
    #  Low‑level helpers
    # ---------------------------------------------------------------------
    def _aa_weight(self, deg: int) -> float:
        """Return 1/log(deg) with safe handling of deg <= 1."""
        if deg <= 1:
            return 1.0  # treat isolated or leaf node as highly informative
        return 1.0 / np.log(deg)

    def _normalized_adamic_adar(self, i: int, j: int) -> float:
        """Normalised Adamic–Adar index between i and j (cf. Eq. 2 in the note)."""
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
        """Nodes at distance exactly 2 from i (friends‑of‑friends) not yet connected."""
        dist2 = set()
        for nbr in self.G.neighbors(i):
            dist2.update(self.G.neighbors(nbr))
        dist2.discard(i)
        dist2.difference_update(self.G.neighbors(i))
        return list(dist2)

    # ---------------------------------------------------------------------
    #  Step 1 – Recommendation
    # ---------------------------------------------------------------------
    def _recommendations(self) -> Dict[int, int]:
        """Return mapping user -> recommended new friend (one per user)."""
        rec: Dict[int, int] = {}
        for i in range(self.N):
            if i in (self.Z_PLUS, self.Z_MINUS):  # zealots get no recs
                continue
            candidates = self._candidate_set(i)
            if not candidates:
                continue
            scores = []
            deg_i = self.G.degree[i]
            for j in candidates:
                aa = self._normalized_adamic_adar(i, j)
                pop = self.G.degree[j] / max(1, self.N - 1)
                hom = 1.0 - abs(self.w[i] - self.w[j])
                score = (
                    self.alpha1 * aa + self.alpha2 * pop + self.alpha3 * hom
                )
                scores.append(score)
            j_star = candidates[int(np.argmax(scores))]
            rec[i] = j_star
        return rec

    # ---------------------------------------------------------------------
    #  Step 2 – Acceptance of proposed edges
    # ---------------------------------------------------------------------
    def _acceptance_probability(self, i: int, j: int, cn_ij: int) -> float:
        """Logistic acceptance probability P_acc(i→j)."""
        s_i = self.s[i]
        deg_i = max(0, self.G.degree[i])

        # Averaged neighbour opinions and alignment ρ_i
        if deg_i:
            avg_friend_op = np.mean([self.w[k] for k in self.G.neighbors(i)])
            rho_i = 1.0 - abs(self.w[i] - avg_friend_op)
        else:
            avg_friend_op = 0.5
            rho_i = 0.0

        # Estimated opinion difference via mutual friends
        if cn_ij:
            mutuals = set(self.G.neighbors(i)) & set(self.G.neighbors(j))
            est_w_j = np.mean([self.w[k] for k in mutuals])
        else:
            est_w_j = avg_friend_op  # default guess
        d_hat = abs(self.w[i] - est_w_j)

        x = (
            self.theta0
            - self.theta1 * s_i
            + self.theta2 * rho_i * cn_ij / (deg_i + 1)
            + self.theta3 * (1.0 - d_hat)
            + self.theta4 * self.EI
        )
        return 1.0 / (1.0 + np.exp(-x))

    def _add_edges_from_recommendations(self, rec: Dict[int, int]):
        for i, j in rec.items():
            if self.G.has_edge(i, j):
                continue  # already friends
            cn_ij = len(set(self.G.neighbors(i)) & set(self.G.neighbors(j)))
            p_acc = self._acceptance_probability(i, j, cn_ij)
            if self.rng.random() < p_acc:
                self.G.add_edge(i, j)

    # ---------------------------------------------------------------------
    #  Step 3 – Opinion update (tapered DeGroot)
    # ---------------------------------------------------------------------
    def _update_opinions(self):
        new_w = self.w.copy()
        for i in range(self.N):
            if i in (self.Z_PLUS, self.Z_MINUS):
                continue  # zealots unchanged
            deg_i = self.G.degree[i]
            if deg_i == 0:
                continue  # isolated node keeps opinion
            avg_neighbor = np.mean([self.w[j] for j in self.G.neighbors(i)])
            taper = 1.0 - abs(2 * self.w[i] - 1) ** self.gamma_taper
            delta = (1.0 - self.s[i]) * taper * (avg_neighbor - self.w[i])
            new_w[i] += delta
        self.w = np.clip(new_w, 0.0, 1.0)

    # ---------------------------------------------------------------------
    #  Step 4 – Prune discordant edges
    # ---------------------------------------------------------------------
    def _tolerance(self, i: int) -> float:
        return np.clip(self.delta0 + self.gamma_tol * self.s[i], 0.0, 1.0)

    def _prune_edges(self):
        to_remove = []
        for i, j in self.G.edges():
            d_ij = abs(self.w[i] - self.w[j])
            for u, v in ((i, j), (j, i)):
                delta_u = self._tolerance(u)
                if d_ij <= delta_u:
                    p_cut = 0.0
                else:
                    p_cut = (d_ij - delta_u) / (1.0 - delta_u)
                if self.rng.random() < p_cut:
                    to_remove.append((i, j))
                    break  # once cut by one side, stop checking
        self.G.remove_edges_from(to_remove)

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------
    def step(self):
        """Advance the simulation by one discrete time step."""
        # 1. recommendation
        rec = self._recommendations()

        # 2. acceptance decision
        self._add_edges_from_recommendations(rec)

        # 3. opinion update
        self._update_opinions()

        # 4. prune edges
        self._prune_edges()

        # 5. evolve external influence EI_t (simple AR(1))
        self.EI = self.rho_EI * self.EI + self.sigma_EI * self.rng.normal()

        self.t += 1

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    def spectral_gap(self) -> float:
        if self.G.number_of_edges() == 0:
            return 0.0
        L = nx.normalized_laplacian_matrix(self.G).todense()
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()
        return float(eigvals[1])  # λ₂

    def average_opinion(self) -> float:
        return float(self.w.mean())

    def run(self, T: int, record_gap: bool = True) -> List[float]:
        """Run T steps, optionally recording spectral gap each time."""
        gaps = []
        for _ in range(T):
            self.step()
            if record_gap:
                gaps.append(self.spectral_gap())
        return gaps


if __name__ == "__main__":
    # Minimal usage example
    model = PolarizationModel(N=200, rng=np.random.default_rng(42))
    gaps = model.run(T=50)
    print("Final spectral gap:", gaps[-1])
    print("Average opinion:", model.average_opinion())
