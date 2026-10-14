"""Synchronous single-agent PUCT over budgeted, recoverable search domains."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Literal

from pydantic import Field

from jevgo.core import (
    BudgetExceeded, Candidate, EnvironmentFailure, FrozenModel, InvalidCandidate,
    RunConfig, SearchDomain, SearchState, Usage, UsagePhase,
)


class RootEdgeStats(FrozenModel):
    candidate: Candidate
    P: float = Field(ge=0, le=1, allow_inf_nan=False)
    N: int = Field(ge=0)
    W: float = Field(allow_inf_nan=False)
    Q: float = Field(allow_inf_nan=False)
    partial_visits: int = Field(ge=0)


class MCTSResult(FrozenModel):
    """No snapshots: safe to serialize as search diagnostics.

    simulations counts attempted simulations, including interrupted ones.
    Partial visits contain only observed truncated returns, with no assertion
    about the missing future value (and no lower-bound claim for signed rewards).
    """
    candidate: Candidate | None = None
    rootstats: tuple[RootEdgeStats, ...] = ()
    simulations: int = Field(default=0, ge=0)
    completed_simulations: int = Field(default=0, ge=0)
    partial_simulations: int = Field(default=0, ge=0)
    stop_reason: Literal["simulation_limit", "budget_exhausted", "terminal", "no_candidates"]
    budget_resources: tuple[str, ...] = ()


@dataclass
class _Node:
    state: SearchState
    edges: list[_Edge] | None = None


@dataclass
class _Edge:
    candidate: Candidate
    N: int = 0
    W: float = 0.0
    partial_visits: int = 0
    reward: float = 0.0
    child: _Node | None = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


class MCTS:
    """PUCT tree search followed by real, highest-prior greedy rollouts.

    max_depth bounds selected/expanded TREE edges. rollout_depth is an
    independent bound on additional transitions from that leaf. Cached tree
    transitions reuse previously observed rewards and snapshots; all new
    transitions are executed by the domain, which alone owns restore/step costs.
    Every call builds a fresh tree; ties preserve candidate input order.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    @staticmethod
    def _check_budget(domain: SearchDomain) -> None:
        domain.ledger.ensure_available(Usage(), phase=UsagePhase.SEARCH)

    @staticmethod
    @contextmanager
    def _timed(domain: SearchDomain, source: str):
        started, prior_wall = perf_counter(), domain.ledger.used.wall_seconds
        metadata, failed = {}, False
        try:
            yield metadata
        except Exception:
            failed = True
            raise
        finally:
            overhead = max(0, perf_counter() - started
                           - (domain.ledger.used.wall_seconds - prior_wall))
            domain.ledger.record(Usage(wall_seconds=overhead), source=source,
                                 phase=UsagePhase.SEARCH, failed=failed, metadata=metadata)

    @staticmethod
    def _normalize(candidates) -> tuple[Candidate, ...]:
        groups = {}
        for candidate in candidates:
            if not isinstance(candidate, Candidate):
                raise InvalidCandidate("SearchDomain.candidates must return Candidate objects")
            groups.setdefault(candidate.action.key(), []).append(candidate)
        masses = [math.fsum(candidate.prior for candidate in group) for group in groups.values()]
        total = math.fsum(masses)
        return tuple(group[0].model_copy(update={
            "prior": mass / total if total else 1 / len(groups),
            "sample_count": sum(candidate.sample_count for candidate in group),
        }) for group, mass in zip(groups.values(), masses))

    def _expand(self, domain: SearchDomain, node: _Node) -> None:
        self._check_budget(domain)
        candidates = self._normalize(domain.candidates(node.state, phase=UsagePhase.SEARCH))
        node.edges = [_Edge(candidate) for candidate in candidates]

    def _select(self, edges: list[_Edge]) -> _Edge:
        parent_visits = sum(edge.N for edge in edges)
        scale = self.config.c_puct * math.sqrt(max(1, parent_visits))
        return max(edges, key=lambda edge: edge.Q + scale * edge.candidate.prior / (1 + edge.N))

    def _rollout(self, domain: SearchDomain, node: _Node, rewards: list[float]) -> float:
        state = node.state
        for depth in range(self.config.rollout_depth):
            if state.terminal:
                return 0.0
            self._check_budget(domain)
            if depth == 0 and node.edges is not None:
                candidates = tuple(edge.candidate for edge in node.edges)
            else:
                candidates = self._normalize(domain.candidates(state, phase=UsagePhase.SEARCH))
            if not candidates:
                break
            candidate = max(candidates, key=lambda item: item.prior)
            first = depth == 0
            if first:
                domain.ledger.ensure_available(Usage(rollouts=1), phase=UsagePhase.SEARCH)
            before_steps = domain.ledger.used.env_steps
            returned = False
            try:
                transition = domain.transition(state, candidate, phase=UsagePhase.SEARCH)
                returned = True
            finally:
                # Preflight before execution; record only a started real rollout.
                # An action followed by a failed snapshot still counts.
                if first and (returned or domain.ledger.used.env_steps > before_steps):
                    domain.ledger.record(Usage(rollouts=1), source="mcts.rollout",
                                         phase=UsagePhase.SEARCH, failed=not returned)
            rewards.append(transition.reward)
            state = transition.state
        if state.terminal:
            return 0.0  # Incoming terminal reward already belongs to its edge.
        self._check_budget(domain)
        return domain.evaluate(state, phase=UsagePhase.SEARCH).value

    def _simulate(self, domain: SearchDomain, root: _Node, path: list[_Edge],
                  rollout_rewards: list[float]) -> float:
        node = root
        for _ in range(self.config.max_depth):
            if node.state.terminal:
                return 0.0
            self._check_budget(domain)
            if node.edges is None:
                self._expand(domain, node)
            if not node.edges:
                break
            edge = self._select(node.edges)
            if edge.child is None:
                transition = domain.transition(node.state, edge.candidate, phase=UsagePhase.SEARCH)
                edge.reward, edge.child = transition.reward, _Node(transition.state)
                path.append(edge)
                node = edge.child
                break
            path.append(edge)
            node = edge.child
        if node.state.terminal:
            return 0.0
        return self._rollout(domain, node, rollout_rewards)

    def _backup(self, path: list[_Edge], rollout_rewards: list[float], tail: float,
                *, partial: bool) -> None:
        value = tail
        for reward in reversed(rollout_rewards):
            value = reward + self.config.gamma * value
        updates = []
        for edge in reversed(path):
            value = edge.reward + self.config.gamma * value
            updated_w = edge.W + value
            if not math.isfinite(value) or not math.isfinite(updated_w):
                raise EnvironmentFailure("Non-finite MCTS return; no path statistics were updated")
            updates.append((edge, updated_w))
        # Validate the full path before applying any visit/value update.
        for edge, updated_w in updates:
            edge.N += 1
            edge.W = updated_w
            edge.partial_visits += int(partial)

    def search(self, domain: SearchDomain, root: SearchState) -> MCTSResult:
        domain.capabilities.require_branching()  # Before candidates, costs, or mutations.
        if root.terminal:
            return MCTSResult(stop_reason="terminal")
        with self._timed(domain, "mcts.search"):
            result = self._search(domain, root)
        if domain.ledger.over_budget:
            result = result.model_copy(update={"stop_reason": "budget_exhausted",
                                               "budget_resources": domain.ledger.over_budget})
        return result

    def _search(self, domain: SearchDomain, root: SearchState) -> MCTSResult:
        tree = _Node(root)
        try:
            with self._timed(domain, "mcts.expand_root"):
                self._expand(domain, tree)
        except BudgetExceeded as error:
            return MCTSResult(stop_reason="budget_exhausted", budget_resources=error.resources)
        if not tree.edges:
            return MCTSResult(stop_reason="no_candidates")
        attempts = completed = partial = 0
        reason, resources = "simulation_limit", ()
        for _ in range(self.config.simulations):
            try:
                self._check_budget(domain)
            except BudgetExceeded as error:
                reason, resources = "budget_exhausted", error.resources
                break
            attempts += 1
            path, rollout_rewards = [], []
            with self._timed(domain, "mcts.simulation") as metadata:
                try:
                    tail = self._simulate(domain, tree, path, rollout_rewards)
                except BudgetExceeded as error:
                    # Unknown future is omitted, explicitly marked as partial.
                    self._backup(path, rollout_rewards, 0.0, partial=True)
                    partial += 1
                    reason, resources = "budget_exhausted", error.resources
                    metadata["partial"] = True
                else:
                    self._backup(path, rollout_rewards, tail, partial=False)
                    completed += 1
                    metadata["partial"] = False
                metadata.update(tree_depth=len(path), rollout_steps=len(rollout_rewards))
            if reason == "budget_exhausted":
                break
            if domain.ledger.over_budget:
                reason, resources = "budget_exhausted", domain.ledger.over_budget
                break
        selected = max(tree.edges, key=lambda edge: (edge.N, edge.Q)).candidate
        stats = tuple(RootEdgeStats(candidate=edge.candidate, P=edge.candidate.prior,
                                   N=edge.N, W=edge.W, Q=edge.Q,
                                   partial_visits=edge.partial_visits) for edge in tree.edges)
        return MCTSResult(candidate=selected, rootstats=stats, simulations=attempts,
                          completed_simulations=completed, partial_simulations=partial,
                          stop_reason=reason, budget_resources=resources)
