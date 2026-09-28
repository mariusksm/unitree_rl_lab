# Anleitung: Bewegungsdaten konvertieren, prüfen und abspielen

Mit dieser Anleitung prüfst du Schritt für Schritt, ob ein Mocap-Clip richtig eingelesen und richtig abgespielt wird.
Grundlage ist der Fix vom 2026-09-28: Früher landeten Fingergelenke der linken Hand im rechten Arm (Review-Fund C1,
siehe `context/REVIEW.md`).

## Hintergrund

Die GMR-`.pkl`-Dateien haben 53 Gelenkspalten. Die Hände liegen **zwischen** den Armen:

```
0-11 Beine | 12-14 Hüfte | 15-21 linker Arm | 22-33 linke Hand | 34-40 rechter Arm | 41-52 rechte Hand
```

Die Konvertierungsskripte nehmen jetzt die Spalten 0–21 und 34–40. Alle `.npz`, die vor dem Fix erzeugt wurden,
waren kaputt; der rechte Arm stand darin starr im Torso. Sie sind gelöscht.

## Aktueller Stand der Dateien

| Was | Pfad |
|---|---|
| Quell-Aufnahmen (korrekt) | `pkl_isaac_lab_fixed_27_07_26/*.pkl` (150 Stück), `pkl_isaac_lab_17_08_26/*.pkl` (7 Stück) |
| Trainings-Clip (neu, geprüft) | `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/robots/g1_29dof/phase1/motions/143_merged_filtered.npz` |
| Weitere konvertierte Clips | noch keine; `pkl_to_npz.py` legt sie standardmäßig in `<pkl-ordner>/npz/` ab |
| Alte Modelle (unbrauchbar) | `logs/rsl_rl/unitree_g1_29dof_ballcatch_phase1/2026-08-31_*` |

Phase 1 und der Unified-Task lesen automatisch die alphabetisch erste `*.npz` in `motions/` ein.

## Vorbereitung

```bash
cd ~/Dokumente/Repositories/unitree_rl_lab
conda activate env_isaaclab_downgrade
P=pkl_isaac_lab_fixed_27_07_26/143_merged_filtered.pkl
M=source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/robots/g1_29dof/phase1/motions
```

## 1. Rohdaten prüfen

Dieser Schritt braucht kein Isaac Sim und dauert nur Sekunden.

```bash
python scripts/mimic/check_motion.py --pkl $P
```

**Erwartet:** Die Tabelle zeigt alle 29 Körpergelenke. Beide Arme haben einen deutlichen Bewegungsumfang (links etwa
4,8 rad, rechts etwa 5,3 rad), und es erscheint keine Warnung „one arm is almost static“.

## 2. Trainings-Clip prüfen (Einlesen)

```bash
python scripts/mimic/check_motion.py --pkl $P --npz $M/143_merged_filtered.npz
```

**Erwartet:** In jeder Zeile `maxdiff 0.000`, kein `MISMATCH`, und am Ende `RESULT: OK — npz matches pkl`.

## 3. Trainings-Clip visuell abspielen

Dafür brauchst du ein Fenster, also **ohne** `--headless`.

```bash
python scripts/mimic/replay_npz.py -f $M/143_merged_filtered.npz
```

**Erwartet:**
- Beide Arme bewegen sich.
- Der rechte Arm steckt nicht im Torso.
- Der linke Arm hebt sich beim Fangen auf etwa Brusthöhe (zwischen ca. 1,5 und 2,0 s im Clip).

Achte außerdem auf die **Beine**, sie waren beim Review auffällig (Hüfte ca. +0,36 rad bei fast gestrecktem Knie):
- Stehen die Füße sauber auf dem Boden?
- Rutschen sie, oder stecken sie im Boden?

## 4. Kurzer Phase-1-Lauf (Einlesen im Task)

Die Logs landen in `/tmp`, dein `logs/` bleibt sauber:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1 --num_envs 16 --max_iterations 10 --experiment_name /tmp/smoke_p1
```

**Erwartet:**
- Es erscheint keine Warnung „falling back to the Gangnam-style PLACEHOLDER“.
- `/tmp/smoke_p1/<timestamp>/params/env.yaml` enthält `motion_file: …/motions/143_merged_filtered.npz`.
- `Episode_Reward/undesired_contacts` ist nahe 0, weil der rechte Arm nicht mehr in den Torso drückt.

## 5. Einen anderen Clip konvertieren und einsetzen

So gehst du vor, wenn du eine andere Aufnahme verwenden willst (hier Beispiel `72`):

```bash
P=pkl_isaac_lab_fixed_27_07_26/72_merged_filtered.pkl
python scripts/mimic/pkl_to_npz.py -i $P --headless          # schreibt nach pkl_isaac_lab_fixed_27_07_26/npz/
python scripts/mimic/check_motion.py --pkl $P --npz pkl_isaac_lab_fixed_27_07_26/npz/72_merged_filtered.npz
python scripts/mimic/replay_npz.py -f pkl_isaac_lab_fixed_27_07_26/npz/72_merged_filtered.npz
```

> **Achtung:** `pkl_to_npz.py` hängt nach `[DONE]` beim Beenden von Isaac Sim. Die Datei ist dann schon vollständig
> geschrieben, du kannst mit **Ctrl+C** abbrechen.

Wenn alles passt, tauschst du den Clip in `motions/` aus. Es darf nur **eine** `.npz` dort liegen, sonst wird die
alphabetisch erste genommen:

```bash
mv $M/143_merged_filtered.npz pkl_isaac_lab_fixed_27_07_26/npz/     # alten Clip aufheben
cp pkl_isaac_lab_fixed_27_07_26/npz/72_merged_filtered.npz $M/
```

Um alle Aufnahmen eines Ordners auf einmal zu konvertieren, gib den Ordner statt einer Datei an:

```bash
python scripts/mimic/pkl_to_npz.py -i pkl_isaac_lab_fixed_27_07_26 --headless
```

Bereits vorhandene `.npz` werden übersprungen, mit `--overwrite` neu erzeugt.

## Danach

1. **Phase 1 neu trainieren** (auf dem korrigierten Clip):
   ```bash
   ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1
   ```
2. **Phase 2 aus dem neuen Phase-1-Checkpoint fine-tunen.** Dabei wird nur der Actor geladen, die Rausch-Std wird
   gesetzt, und der Lauf loggt in den Phase-2-Ordner:
   ```bash
   ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 --resume \
     --resume_experiment unitree_g1_29dof_ballcatch_phase1 --load_run <neuer_run> \
     --finetune --finetune_std 0.2
   ```
3. Für den Unified-Task vorher `CATCH_MOTION_TIME` in `unified/ball_catch_env_cfg.py` auf den Fang-Moment des Clips
   setzen. Den Moment bestimmst du mit Schritt 3.
