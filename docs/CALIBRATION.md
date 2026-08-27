# Calibration notes

Every numeric prior in the model is recorded here with the reasoning behind it.
The goal is not precision — it cannot be, since no public data set gives
per-control efficacy in hospital estates — but **auditability**: a reviewer who
disagrees with a value can see exactly what it was based on and change it.

## 1. Zone parameters

**Restoration times** come from the publicly reported recovery durations of the
2024 incidents. Ascension's electronic health record outage ran for
approximately four weeks across 142 hospitals; Change Healthcare's clearing-house
disruption ran for weeks into months; Synnovis pathology services were degraded
for an extended period after the June 2024 attack. The EHR core is therefore
given a 240-hour base restoration time, rising to roughly 528 hours when backups
are also destroyed. Billing is given 216 hours to reflect the clearing-house
case; pathology 168 hours.

**Outage cost per hour** is set so that a multi-week EHR outage lands in the tens
of millions, consistent with reported recovery costs for large providers.

**PHI record counts** total 2.29 million across the estate, which is the order of
magnitude of a large multi-site provider. Change Healthcare's 192.7 million is
deliberately *not* used: that is a national clearing-house, not a provider, and
including it would distort the reference architecture.

**Base detection hazards** `d_i` are set so that, with no dedicated monitoring, a
single compromised zone takes on the order of a week to be noticed, and a broad
compromise is noticed within days. This is the regime implied by pre-2020 dwell
time reporting.

## 2. Edge probabilities

`p_base` is the probability that a competent adversary can traverse the step
**with no mitigating control present at all**. Values are high by design: in a
flat network with no segmentation, no endpoint detection, no multi-factor
authentication and no patching, lateral movement is close to routine. Anchors:

* `internet -> email` at 0.78 — a sustained phishing campaign against a large
  workforce lands.
* `internet -> remote_access` (credential abuse) at 0.62 — the Change Healthcare
  intrusion is publicly reported to have entered through a Citrix remote-access
  portal that lacked multi-factor authentication.
* `email -> corp_workstations` at 0.86 — Ascension's investigation concluded an
  employee downloaded a malicious file.
* `identity -> *` at 0.80–0.90 — domain compromise generalises to nearly
  everything joined to the domain.
* `vendor_gateway -> lab_lis` at 0.74 — the Synnovis pathology compromise reached
  NHS trusts through a supplier relationship.

`tau` is a mean traversal time in hours, set so that a full campaign to the EHR
core takes days rather than hours or months.

## 3. Control efficacy

`ETA[control][edge_kind]` is the fractional reduction in traversal probability at
full coverage. Three rules constrain the table:

1. **No entry exceeds 0.80.** No control eliminates a class of attack. This keeps
   residual risk strictly positive for every portfolio, which is both realistic
   and necessary for the loss-exceedance curves to behave.
2. **Efficacy is assigned only where a causal mechanism exists.** Immutable
   backup has zero prevention efficacy against every edge kind except
   `backup_tamper`, because it does not stop an adversary from moving — it only
   preserves the ability to recover.
3. **The largest single entry is MFA against credential abuse (0.78)**, which is
   the mechanism the public reporting on Change Healthcare points at most
   directly.

Detection gains, containment gains and recovery gains are separate tables because
these are distinct effects: SIEM does not prevent lateral movement, it shortens
the time before someone notices it.

## 4. Cost model

Annual cost at full coverage is a rough programme cost for a large provider,
including licensing, staff and integration. Cost is superlinear in coverage with
exponent 1.35, because the last few percentage points of coverage in a hospital
estate — legacy imaging modalities, unmanaged biomedical devices, clinical
workflow exceptions — are disproportionately expensive. This exponent is a
judgement, and the greedy frontier in `scripts/07_frontier.py` is sensitive to
it; the ordering of controls is not.

## 5. Loss model

Breach cost uses `a * R^b` with `b = 0.75`, i.e. sub-linear in record count,
matching the well-documented volume effect in reported breach costs. `a = 373` is
chosen so that a 100,000-record breach costs roughly 2.1 million currency units.

Ransom payment probability is conditioned on whether recovery infrastructure
survived: 0.62 when backups were destroyed, 0.11 when they were not.

## 6. Campaign frequency

`lambda = 1.5` serious targeted campaigns per year against one large provider.
This is the weakest-grounded parameter in the model and it scales the annualised
loss expectancy and the return on security investment linearly. It does **not**
affect the ordering of portfolios or any ablation conclusion, all of which are
computed on per-campaign loss.

## 7. What was *not* tuned

The five face-validity targets in `scripts/08_calibration.py` were fixed before
the model parameters were frozen, and no parameter was adjusted afterwards to
improve agreement. Four of the five are matched. The fifth — the share of landed
attacks that disrupt care delivery — comes out below the range reported in the
epidemiological literature, and is reported as a shortfall.
