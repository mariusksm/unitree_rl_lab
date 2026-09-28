# CLAUDE.md

Fork von `unitree_rl_lab` (Isaac Lab 2.3.0 / Isaac Sim 5.1.0) für die Projektgruppe **H-ReACT**:
Ein Unitree G1 (29 DoF) soll einen zugeworfenen Ball fangen. Die Projektarbeit liegt in
`source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/` und `scripts/mimic/`.

## Zuerst lesen

**`unitree_rl_lab_spec.md`** ist die zentrale Spezifikation und Stand-Aufnahme: Architektur,
Tasks, Observation-Vertrag, Datenpipeline, Trainingsstand, offene Punkte, Änderungslog.
Vor jeder Arbeit lesen. Hintergrundmaterial wie Antrag, Protokolle und Pläne liegt in `context/`.

## Pflicht: Spec-Dokument pflegen

Jede **substanzielle Änderung** muss in `unitree_rl_lab_spec.md` festgehalten werden, und zwar in
dieser Reihenfolge:

1. **Zuerst die Spezifikation anpassen:** die betroffenen Kapitel (Tasks, Observation-Layout, Rewards,
   Pipeline, Status, offene Punkte …) so aktualisieren, dass sie den neuen Ist-Zustand beschreiben.
   Die Spec beschreibt immer den *aktuellen* Stand und keine Historie.
2. **Dann einen Log-Eintrag anhängen:** im Kapitel „Änderungslog“ am Ende des Dokuments einen neuen
   Eintrag **unten** ergänzen (`### YYYY-MM-DD: Kurztitel` + Stichpunkte: was, warum, betroffene Dateien
   bzw. Kapitel). Bestehende Log-Einträge werden nie verändert oder gelöscht, damit die Historie erhalten bleibt.

Als substanziell gilt u. a.: neue oder geänderte Tasks, Env-Configs, Rewards, Observations,
Terminierungen, Curricula, PPO-Hyperparameter, Änderungen an Datenformat oder Pipeline-Skripten,
neue Trainingsergebnisse bzw. Checkpoints mit Aussagekraft, Architekturentscheidungen, Deploy- und
Sim2Real-Arbeit. Nicht substanziell: reine Formatierung, Tippfehler, Kommentare.

## Wichtige Invarianten (Details in der Spec)

- Der Observation-Slot-Vertrag gilt über Phase 1 / Phase 2 / Unified hinweg: Änderungen in allen Tasks
  und in beiden Gruppen (Policy/Critic) spiegeln.
- Ball- und Hand-Observations im Yaw-Frame des Roboters, niemals Weltkoordinaten.
- Action-Skalierung überall `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE`.

## Umgebung

- Python: `/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python`
- Training: `./unitree_rl_lab.sh -t --task <Task>` · Play: `-p` · Liste: `-l`
