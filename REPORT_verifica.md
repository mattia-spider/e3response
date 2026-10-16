# Verifica architettura deriv2nd e igiene dei risultati

Branch `verifica-deriv2nd` (e3response). Nessun config o codice esistente è stato modificato; tutto
il nuovo sta in `tests/verifica/`, `scripts/`, `src/e3response/nodewise_product.py` e
`configs/model/nequip_nmr_deriv2nd_muB.yaml`.

## Riassunto

1. **Ipotesi confermata sui pesi random (float64).** Nel deriv2nd σ è esattamente 0 (0.0, non solo
   piccolo) per atomo isolato, HCN, C2H2 e per il blocco xy delle planari; deriv1st e direct danno
   σ ≠ 0 negli stessi casi. Sulle planari sopravvive solo zz.
2. **Con μ⊗B negli attributi gli zeri spariscono** (`deriv2nd_muB`): σ ≠ 0 su tutti i casi, ancora
   equivariante (1e-15) e coerente con le differenze finite (4e-10). Non so ancora se la val scende:
   richiede la run sul cluster (non fatta qui).
3. **Equità del confronto, problemi emersi:** nessun modello normalizza σ; il direct usa SH di bordo
   fisse a `0e+1o+2e` e readout a `ell_max=2` anche con l=3; le loss differiscono (direct: energia;
   deriv1st: L2 su B_ind senza energia; deriv2nd: energia + L2 su B_ind). Il peso dell'energia (1.0
   su Ha²) è ~4e-7 volte il termine dei tensori: l'energia è di fatto non pesata.
4. **`test_Z0` non è F:** è la specie 0 (atomo di padding) entrata in `from_data.atomic_numbers`
   nelle run precedenti al fix reax del 2026-10-01.
5. **Non verificabile in locale:** checkpoint e risultati della griglia Jean Zay, `summarize_grid.py`
   e i config `escale`/`deriv2nd_noE`/`noE` non sono nel repo. Quali run rifare: vedi sotto.

Legenda: ✅ fatto · ⚠️ fatto con riserve · ❌ non fatto / non fattibile in locale.
Ambiente: `mlplay`, CPU, JAX 0.6.2, float64 per i test B. Il test B è stato **ridimensionato** (sotto).

---

## A. Ispezione del codice

**A1 ✅** `tensorial/gcnn/nequip.py`, `_message_passing.py`.
- Self-connection: `e3j.flax.Linear(..., num_indexed_weights=num_species)(node_species, node_features)`.
  L'indice è il **solo intero di specie**; i `node_features` entrano linearmente. `attributes`
  (con μ e B) non viene riusato dopo il primo `NodewiseLinear`.
- Convoluzione: `tensor_product(node_feats[senders], edge_SH)` pesato da un MLP radiale: lineare nelle
  feature, l'altro fattore è solo geometria.
- Gate: `e3j.gate` con silu su 0e, tanh su 0o; i gate dei non-scalari sono 0e extra creati da
  `linear_down`. Gli 0o (parità `o` negli hidden irreps) sono solo attivati con tanh (dispari).
- Domanda chiave: **nessun punto, oltre al gate, moltiplica due feature dipendenti dai campi.**
  Argomento (non testato separatamente): un 0e lineare nei campi si ottiene solo da `1o ⊗ Y1`, con 1o
  lineare nei campi = μ×r̂; da qui `(μ×r̂_ij)·r̂_ik = μ·(r̂_ij×r̂_ik)`, cioè μ·a con **a vettore assiale
  geometrico**. Il gate lo moltiplica per un vettore lineare in B: i termini bilineari sono
  `(μ·a)(…B…)`, come nell'ipotesi. Per atomo isolato/lineare/planare a manca o è ∥ n.

**A2 ✅** `NodewiseEncoding` (= `NodewiseEmbedding`) concatena nell'ordine del dict
`[one-hot specie (0e), B_ext (1e), μ (1e)]` in `nodes.attributes`; i globals sono broadcast sui nodi.
Il primo `NodewiseLinear(field: attributes)` scrive `nodes.features`; `attributes` non è più letto.

**A3 ⚠️** `e3response/magnetic.py`.
- `InducedMagneticField`: `B_ind = ∂E/∂μ` per nodo con `gcnn.diff(..., mode="rev")`, μ = zeros(3)
  broadcast a (N,3) e differenziata per nodo; valutata in μ=0.
- `MagneticShieldingTensor`: `σ_{k,γα} = ∂B_ind,γ / ∂B_α` con `jacfwd`, B_ext = zeros(3) condiviso dal
  batch (esatto perché i grafi sono indipendenti; verificato in `tests/test_magnetic.py`).
- Segno: deriv2nd usa `+∂E/∂μ`. Il segno fisico (E = −μ·B) è assorbito dall'apprendimento.
- Dataset: `qm9_nmr.py` legge `XX,YX,ZX,XY,YY,ZY,XZ,YZ,ZZ` e fa `reshape(3,3)` per righe, quindi
  `M[0,1]` = etichetta "YX". Se Gaussian intende "YX" come σ_yx, `M = σᵀ` rispetto alla convenzione del
  modello (σ_ij = ∂B_i/∂B_j). **Non verificato.** Non è un problema di equità (tutti i modelli
  imparano lo stesso `M`), ma cambia il segno fisico della parte 1e.
- B6 conferma che l'autodiff è la derivata seconda corretta di E (vedi B6).

**A4 ⚠️** Normalizzazioni (config del working tree):

| | direct | deriv1st | deriv2nd |
|---|---|---|---|
| shift/scale su σ | no | no | no |
| `per_species_rescale` | solo energia (scales=1.0) | — | solo energia (scales=1.0) |
| edge SH | `0e+1o+2e` fisse | `ell_max` | `ell_max` |
| readout | `ell_max=2`, parità e | 4×1e | 4×0e |
| loss | energia + σ | σ + L2(B_ind), **no energia** | energia + σ + L2(B_ind) |

Il confronto non è quindi del tutto equo (SH/readout del direct a l≤2, energia assente nel deriv1st,
L2 su B_ind in entrambi i deriv). Nessun σ è normalizzato in nessun modello (a parità di trattamento).

---

## B. Test numerici sull'ipotesi (pesi random)

Script: `tests/verifica/b_tests.py`, `common.py`. Risultati: `results_B_quick.json`,
`results_B_mul4.json`, `results_muB.json`. Valori **relativi** al max|σ| della molecola generica con la
stessa rete; molecole costruite a mano (HCN, C2H2, CO2, H2CO, benzene, un'etanolo-like).

**Copertura effettiva (ridimensionata):**
- deriv2nd: l=2, mul=4, n=2 e n=3, 3 seed per ciascuno.
- deriv1st e direct: l=2, mul=4, n=2, 2 seed (run rapida).
- **Non eseguiti:** l=3, mul=8, B7. La matrice completa è stata interrotta su richiesta.

| (max su seed, relativo) | deriv2nd n=2 | deriv2nd n=3 | deriv1st | direct | deriv2nd_muB |
|---|---|---|---|---|---|
| B1 atomo C | **0** | **0** | 2e-3 – 1.3e-1 | 7.5e-2 – 3.4e-1 | 5e-2 – 2.5e-1 |
| B2 HCN | **0** | **0** | 0.44 – 0.71 | 0.85 – 0.94 | 0.79 – 0.89 |
| B2 C2H2 | **0** | **0** | 0.50 – 0.59 | 0.66 – 0.93 | 0.37 – 0.93 |
| B3 H2CO blocco xy | **0** | **0** | 0.40 – 0.62 | 0.81 – 0.83 | 0.84 – 1.1 |
| B3 H2CO zz | 0.11 – 0.28 | ≥ 0.045 | 0.11 – 0.38 | 0.80 – 0.84 | 0.83 – 0.91 |
| B3 xz, yz, zx, zy | 0 | 0 | 0 | 0 | 0 |
| B4 componente minima (9/9 ≠ 0) | 0.17 – 0.20 | ≥ 0.18 | 0.055 – 0.11 | 0.020 – 0.054 | 0.008 – 0.26 |
| B5 equivarianza (rel.) | ≤ 2.4e-15 | ≤ 4.5e-15 | 1e-15 | 1e-15 | 1.4e-15 |
| B5 inversione | 0 | 0 | 0 | 0 | 0 |
| B6 autodiff vs FD (h=1e-3) | 5e-5, 1e-5 | — | — | — | 4.6e-10, 4.0e-10 |

- **B1–B3 ✅ ipotesi confermata:** zero numerico (< 1e-6 relativo; qui 0.0 in float64) per il deriv2nd e
  non per deriv1st/direct. Il criterio del TODO è soddisfatto. Le componenti con indice lungo la normale
  sulle planari sono 0 per tutti i modelli (simmetria di riflessione), perciò il controllo
  discriminante è il blocco xy.
- La riserva rispetto alla formulazione "solo righe/colonne lungo n": nel deriv2nd sopravvive **solo
  zz**, non l'intera riga/colonna n.
- Osservazione: la parte antisimmetrica di σ è ~46–55% della norma nel deriv2nd a init, 13–22% nel
  deriv1st, 0% nel direct: coerente con σ ≈ a⊗b (prodotto di due vettori).
- **B4 ✅** nessun componente nullo per nessun modello su una molecola non planare. Nel deriv2nd_muB
  un seed ha una componente minima di soli 8e-3: ≠ 0 ma piccola.
- **B5 ✅** σ(Rx) = Rσ(x)Rᵀ a ~1e-15; inversione σ invariante (differenza 0).
- **B6 ✅** errore relativo 1e-5–5e-5 per il deriv2nd (consistente con il troncamento O(h²)), 4e-10 per
  il muB (la differenza è soprattutto la scala di σ, ~1e6 volte più grande; l'errore assoluto è
  1e-11–1e-10 in entrambi i casi).
- **B7 ❌ non eseguito.** Dato indicativo: max|σ| a init del deriv2nd originale ≈ 2e-7 – 1e-6 ppm, del
  muB ≈ 0.13–0.19 ppm, di deriv1st/direct ~1e-3–1e-2; il target ha rms ≈ 80 ppm (vedi F3). Lo script
  `b7_scale.py` è pronto ma non è stato lanciato.

---

## C. Analisi dei checkpoint

**C1–C3 ⚠️** Script `scripts/analyze_ckpt.py` (CSV per atomo + `_summary.csv`): errore per elemento ×
irrep (0e/1e/2e, con norma target/predizione della 1e), planarità (autovalore minimo del tensore di
girazione degli atomi pesanti), numero di vicini entro r_max, errore nel piano vs fuori piano per le
planari. Provato solo su una run locale piccola (4 molecole di test); gira su CPU, nessuna dipendenza
dal cluster oltre ai path dei dati (`--data-dir`).

**C4 ❌** I checkpoint indicati (direct l=3 mul=64 n=4; deriv1st n=6; deriv2nd l=3/l=2 mul=32) non sono
in questa macchina. Comando: `python scripts/analyze_ckpt.py --run-dir <run> --split test --out <csv>`.

**Extra ✅ (script)** `scripts/check_sigma_zero.py`: dato un checkpoint deriv2nd, stampa max|σ| su
atomo, HCN, C2H2, H2CO, benzene e molecola generica. Provato su una run deriv1st locale (σ ≠ 0
ovunque, come atteso). È il controllo suggerito per confermare il meccanismo sui pesi allenati.

---

## D. Prototipo di correzione (ridimensionato)

**D1 ✅** `src/e3response/nodewise_product.py` (`NodewiseFieldProduct`) aggiunge agli attributi
`B ⊗ μ` (0e+1e+2e) subito dopo `NodewiseEncoding`, prima del primo `NodewiseLinear`. Legge i due
blocchi 1e di `attributes` (già broadcast). Config: `configs/model/nequip_nmr_deriv2nd_muB.yaml`,
identico al deriv2nd tranne per il layer (parametro opzionale `muB_scale`, default 1.0). Il blocco è
nullo a μ=B=0: l'energia in quel punto non cambia struttura.

**D2 ⚠️** Scala: a init max|σ| ≈ 0.13–0.19 ppm contro ~80 ppm di rms del target. `muB_scale` non è
stato calibrato (dipende dalla run). Se la val resta alta, la scala è il primo sospetto.

**D3 ⚠️** B1–B6 su `deriv2nd_muB` (l=2, mul=4, n=2, 2 seed): vedi tabella B; σ ≠ 0 ovunque, equivariante,
FD ok.

**D4 ❌** Il test di sovra-adattamento è stato scartato nel ridimensionamento. La verifica reale è la
run sul cluster: deriv2nd l=2, mul=16, n=2, r=5, stesso seed/split, solo
`model=nequip_nmr_deriv2nd_muB`. Lettura: val verso ~5 e H/O da decine di ppm a pochi ppm →
ipotesi giusta; val ~20 → il problema è altrove (probabilmente la scala).

---

## E. Igiene dei risultati

**E1 ⚠️** `scripts/summarize_runs.py` legge i valori **risolti** da `config.yaml` di ogni run
(r_max, ell_max, mul, num_layers, pesi della loss, flag `induced_B`, `n_elements`, `has_Z0`, limit,
batch_size) e ogni colonna `test*` del csv. Provato su `logs/train/runs`: 89 complete, 112
incomplete; nessuna colonna di config vuota. `summarize_grid.py` originale non è nel repo, quindi
non l'ho "corretto": questo ne è un sostituto.

**E2 ✅** `test_Z0` è la metrica per la **specie con numero atomico 0**, cioè l'atomo di padding che
finiva in `from_data.atomic_numbers` (es. run con `[0,1,6,7,8,9]`, `n_elements=6`). Non ha atomi reali
→ NaN. Non è F (Z=9). Corretto a monte in reax (commit 82f94fb, 2026-10-01): le run successive non
lo hanno. Le run precedenti avevano anche un tipo di specie in più (n_elements +1), differenza
minore ma presente.

**E3 ⚠️** Lo script marca `status = incomplete` (da escludere dal confronto, file `*_incomplete.csv`)
quando non ci sono metriche `test*` né "Best ckpt path" in `train.log`. Euristica: le run deriv1st
n=7, 8 non sono qui, quindi non l'ho applicata a loro.

**E4 ⚠️** Lo script calcola per ogni run un hash md5 degli indici train/val/test ricostruiti con
`Qm9NmrDataModule.load_split`-equivalente (dataset, limit, frazioni), e riporta quanti hash distinti
ci sono. Riserve: usa `nnx.Rngs(0)` (default di reax); non ho potuto confermare che il trainer non
avanzi quel generatore prima dello split. Su 2 run locali l'hash funziona; confronto reale non fatto.

**E5 ❌** I config `nequip_nmr_escale`, `nequip_nmr_deriv2nd_noE`, `nequip_nmr_noE` non esistono in
`configs/model/`; nulla da documentare in locale.

---

## F. Energia

Script: `tests/verifica/f_energy.py`, risultati `results_F.json` (3000 molecole casuali; fit sull'80%,
test sul 20%).

**F1 ✅** L'energia è l'`SCF Done: E(...) = … A.U.` del log Gaussian, quindi **Hartree** (totale per
molecola, range −585.6 / −270.2 Ha, media −411.2). 1 Ha = 627.509 kcal/mol (`summarize_runs.py`
aggiunge `test_energy_rmse_kcal`).

**F2 ✅** `energy_shifts` = `EnergyContributionLstsq(normalize=True)`: minimi quadrati su conteggi ed
energie normalizzati per numero di atomi (non media). Baseline per specie sul test: **MAE 0.0323 Ha =
20.3 kcal/mol** (RMSE 25.7 kcal/mol); i minimi quadrati sui totali danno lo stesso (20.26 kcal/mol).
Shift (Ha): H −0.615, C −38.079, N −54.753, O −75.230, F −99.874.

**F3 ✅/⚠️** `response_loss(energy=True, nmr_tensors=True)`: `optax.squared_error` su
`globals.energy` (energia **totale per molecola**, non per atomo) e su `nodes.nmr_tensors`; riduzione
`mean` su nodi, poi su grafi, poi sulle 9 componenti; pesi 1.0 e 1.0 sommati direttamente. Quindi
1 Ha² pesa come 1 ppm². Con il solo baseline, MSE energia = 1.68e-3 Ha² contro MSE tensori ≈ 4347
ppm² (6493 ppm² con predizione nulla): rapporto **3.9e-7**. L'interpretazione "Ha/ppm" è corretta,
ma il peso 1.0 rende l'energia praticamente non vincolante rispetto ai tensori.

**F4 ✅** Con `energy: true` (bool) il peso di default è **1.0**; `energy_w` vuoto nella tabella va
letto come 1.0 (e 0 se `energy` è assente/False). Per i tensori vale lo stesso (1.0).

---

## Cosa rifare / prossimi passi

- **Run da fare (cluster):** deriv2nd_muB, l=2, mul=16, n=2, r=5, stesso seed e split del riferimento
  (val 20.98); poi, se funziona, l=3, mul=32, n=4, r=4. Prima, `check_sigma_zero.py` sul checkpoint
  deriv2nd attuale per confermare gli zeri sui pesi allenati.
- Le run deriv2nd originali non sono migliorabili da tuning dell'ottimizzatore: la limitazione è
  strutturale (B1–B3). Se il muB non basta, calibrare `muB_scale`.
- Non confrontare le run `incomplete`; controllare gli hash di split sulla griglia reale.
- Non fatto: B7, B con l=3/mul=8, C4, E3/E5 sui dati reali.
