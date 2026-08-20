"""The preregistered decision rules, as code.

Kept here rather than inline in the analysis scripts so they can be tested
against synthetic inputs with known answers. Verifying the decision path on
synthetic data rather than on a slice of the real run avoids previewing the
result, which is the whole point of fixing the rules in advance.

Thresholds are transcribed from docs/prereg_round1.md and must not be edited to
suit an outcome.
"""

# Experiment 1
E1_SUPPORT = 0.20  # median rho_spatial at or above this, CI excluding 0 -> supported
E1_REFUTE_SPATIAL = 0.10  # below this ...
E1_REFUTE_SEQ = 0.30  # ... while rho_seq is at or above this -> refuted

# Experiment 2
E2_RETENTION = 0.10  # a genuinely spatial head keeps rho_spatial within this
E2_LOSS_GATE = 2.0  # loss ratio above which E2 is a distribution-shift result

# Experiment 4 / 5 (round 2)
E4_SUPPORT = 0.15  # rho_adj at or above this, CI excluding zero -> topology survives
E4_REFUTE = 0.05  # below this, CI excluding E4_SUPPORT -> no topological structure
ADJACENCY_RATE = 0.0115  # measured fraction of causal face pairs sharing an edge

# Experiment 3
E3_INHERITED = 0.5
E3_INDEPENDENT = 0.2

SINK_THRESHOLD = 0.5  # dormant above this, active at or below


def e1_verdict(med_spatial, ci_lo, med_seq):
    if med_spatial >= E1_SUPPORT and ci_lo > 0:
        return "SUPPORTED", "spatial selectivity survives the control"
    if med_spatial < E1_REFUTE_SPATIAL and med_seq >= E1_REFUTE_SEQ:
        return "REFUTED", "attention is recency, not geometry"
    return "INCONCLUSIVE", "at this n"


def e2_verdict(delta_median, loss_ratio):
    if loss_ratio > E2_LOSS_GATE:
        return "GATE_TRIPPED", "re-sorted input is off-distribution; E1 stands as primary"
    if abs(delta_median) <= E2_RETENTION:
        return "STABLE", "spatial selectivity is stable under the re-sort"
    return "FOLLOWS_SORT_KEY", "apparent spatial selectivity follows the sort key"


def e4_verdict(med_adj, ci_lo, ci_hi):
    if med_adj >= E4_SUPPORT and ci_lo > 0:
        return "SUPPORTED", "attention carries topological structure beyond recency and proximity"
    if med_adj < E4_REFUTE and ci_hi < E4_SUPPORT:
        return "REFUTED", "no topological structure; the recency account stands"
    return "INCONCLUSIVE", "at this n"


def e3_verdict(rho):
    if rho >= E3_INHERITED:
        return "INHERITED", "sink head identity survives the modality swap"
    if rho <= E3_INDEPENDENT:
        return "INDEPENDENT", "sinks re-form under the new modality"
    return "PARTIAL", "partial inheritance"
