# unitree_rl_lab: Spezifikation & aktueller Stand

> **Zweck dieses Dokuments:** Schneller, verlässlicher Einstieg für Menschen und AI-Assistenten.
> Beschreibt, *was* dieses Repository im Rahmen der Projektgruppe H-ReACT tut, *wie* es aufgebaut ist
> und *wo* es gerade steht.
> **Pflegeregel:** Bei jeder substanziellen Änderung werden zuerst die betroffenen Kapitel
> angepasst, danach wird am Ende im Kapitel [Änderungslog](#14-änderungslog) ein Eintrag ergänzt (siehe `CLAUDE.md`).
>
> Stand: **2026-09-28**. Pfadangaben sind relativ zum Repo-Root. `BC/` steht als Abkürzung für
> `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/`.

---

## 1. Projektkontext

- **Projektgruppe H-ReACT** (Humanoid Reactive Action & Control Training), TU Dortmund und Fraunhofer IML,
  SoSe 2026 bis WiSe 2026/27.
- **Ziel:** Ein **Unitree G1 (29 DoF)** fängt einen von einem Menschen zugeworfenen Ball.
  Die Ganzkörpersteuerung wird per RL in **Isaac Lab** trainiert, die Referenz liefern menschliche
  Mocap-Daten (VICON Shogun, Manus-Handschuhe). Die Ballposition ist **Ground Truth**
  (in der Simulation aus Isaac, in echt aus VICON).
- **Minimalziele laut Antrag:** (1) Policy fängt in Isaac Sim wiederholt mit GT-Ballposition,
  (2) Einsatz auf dem echten G1 mit **über 50 % Erfolgsrate**, (3) Endbericht, (4) Fachgespräch.
- **Hände:** Laut Antrag wird die 5-Finger-Hand durch eine **statische, 3D-gedruckte Aufnahme** ersetzt.
  Das Robotermodell im Repo hat **keine Hände**. Gefangen wird über die Links `*_wrist_yaw_link`.
- Dieses Repo ist ein Fork von [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
  (Isaac Lab 2.3.0 / Isaac Sim 5.1.0, RSL-RL PPO). Die projektspezifische Arbeit steckt fast vollständig in
  `BC/`, `scripts/mimic/pkl_to_npz.py` und `scripts/pkl_to_csv_without_hands.py`.
- Andere Teile der PG arbeiten in **anderen Repos** (GitLab des Fraunhofer: Mocap-Aufnahme,
  Retargeting mit GMR/SOMA, Eventkamera, LocoMuJoCo, MuJoCo). Hintergrundmaterial wie Antrag,
  Protokolle und Pläne liegt in `context/` (siehe [Kapitel 12](#12-kontextdokumente)).

## 2. Status auf einen Blick (2026-09-28)

| Bereich | Stand |
|---|---|
| Phase 1 (Mimic) auf echten Mocap-Daten | ✅ trainiert: Lauf `2026-08-31_15-06-41`, 1657 Iterationen, Clip `143_merged_filtered.npz`. Reward −1,3 → 89,7, Episodenlänge → 1413 Schritte, `time_out` 86 %. Tracking funktioniert. |
| Phase 2 (RL-Fangen, Transfer aus Phase 1) | ⚠️ Code steht (P0/P1-Fixes), aber **kein Trainingslauf** nach den Fixes (Log-Ordner leer). |
| Unified-Task (Mimic + Fangen gleichzeitig) | ⚠️ Code steht, **nie trainiert**. `CATCH_MOTION_TIME = 2.0` ist ein **Platzhalter**. |
| Fangerfolg in der Simulation | ❌ noch nicht nachgewiesen |
| Hände (Dex3 / Aufnahme) im Modell | ❌ nicht vorhanden (bewusst, siehe Antrag) |
| Rauschen, Latenz und Dropout auf Ball-Observations | ❌ nicht implementiert |
| Sim2Sim (MuJoCo), Deploy, VICON-Livepipeline | ❌ für den Ball-Task nicht begonnen (`deploy/` ist unverändert Unitree-Upstream) |

## 3. Repository-Layout (relevanter Ausschnitt)

```
unitree_rl_lab.sh                      Launcher: -i install, -l list, -t train (headless), -p play
scripts/
  rsl_rl/train.py, play.py, cli_args.py   Upstream + Fix: --experiment_name wird übernommen (für --resume)
  list_envs.py
  pkl_to_csv_without_hands.py          GMR-.pkl → CSV (nur 29 Körpergelenke, 24 Handspalten verworfen)
  mimic/csv_to_npz.py                  CSV → .npz (Isaac-Replay, Upstream)
  mimic/pkl_to_npz.py                  NEU: GMR-.pkl → .npz direkt, ganzer Ordner in einer Isaac-Session
  mimic/replay_npz.py                  .npz visuell prüfen
source/unitree_rl_lab/unitree_rl_lab/
  assets/robots/unitree.py             UNITREE_G1_29DOF_MIMIC_CFG, UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
                                       lokale Pfade UNITREE_MODEL_DIR / UNITREE_ROS_DIR
  tasks/locomotion/, tasks/mimic/      Upstream-Tasks (Velocity, Gangnam, Dance)
  tasks/ball_catching/                 ← PROJEKT-TASK (= BC/)
    agents/rsl_rl_ppo_cfg.py           BasePPORunnerCfg, Phase2PPORunnerCfg
    mdp/                               commands/, observations, rewards, terminations, events, curriculums
    robots/g1_29dof/phase1/            Env-Cfg + motions/ (Drop-in-Ordner für .npz, README)
    robots/g1_29dof/phase2/            Env-Cfg
    robots/g1_29dof/unified/           Env-Cfg
pkl_isaac_lab_fixed_27_07_26/          150 retargetete GMR-.pkl (+ npz/ mit 5 konvertierten), gitignored
pkl_isaac_lab_17_08_26/                7 neuere .pkl (Stand 17.08.), gitignored
logs/rsl_rl/<experiment>/<timestamp>/  Checkpoints model_*.pt, params/, tfevents
deploy/                                C++-Deploy (ONNX, unitree_sdk2), Upstream, für den Ball-Task unangetastet
context/                               PG-Hintergrund (nicht versioniert, siehe Kapitel 12)
```

`.npz`-Dateien sind per `.gitignore` global ausgeschlossen. Der Trainingsclip in `phase1/motions/` ist
deshalb **nicht im Git** und muss auf jeder Maschine separat liegen.

## 4. Umgebung & Befehle

- Python-Env (Marius' Rechner): `/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python`
  (Isaac Lab 2.3.0, Isaac Sim 5.1.0). Der System-Python hat kein numpy.
- Tasks registriert unter: `Unitree-G1-29dof-BallCatch-Phase1`, `-Phase2`, `-Unified`
  (außerdem Upstream: `Unitree-G1-29dof-Velocity`, `-Mimic-Gangnanm-Style`, `-Mimic-Dance-102`, Go2/H1 Velocity).

```bash
./unitree_rl_lab.sh -l                                           # Tasks auflisten
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1  # Training headless
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-BallCatch-Phase1  # Play (lädt letzten Run)
# Phase 2 von Phase-1-Checkpoint (Phase2PPORunnerCfg: init_noise_std=0.2)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 --resume \
  --experiment_name unitree_g1_29dof_ballcatch_phase1 --load_run <timestamp> --checkpoint model_<N>.pt
# Phase 2 from scratch: zusätzlich --agent.policy.init_noise_std=1.0
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Unified # Unified (from scratch, std 1.0)
# Smoke-Test: --num_envs 16 --max_iterations 2
tensorboard --logdir logs/rsl_rl/
```

## 5. Datenpipeline (Mocap → Trainingsclip)

```
VICON Shogun (+Manus) ──(andere Repos: Bereinigung, GMR-Retargeting)──► *_merged_filtered.pkl
  .pkl: root_pos, root_rot (xyzw), dof_pos (53 Spalten = 29 Körper + 24 Hand), fps (meist 30)
      │
      ├─ empfohlen: scripts/mimic/pkl_to_npz.py -i <ordner|datei> [--headless] [-o out] [--output_fps 50]
      │     29 Körpergelenke → Interpolation (lerp/slerp) auf 50 fps → Finite-Differenzen-Geschwindigkeiten
      │     → kinematischer Replay in Isaac Sim → Forward Kinematics aller Bodies → <input>/npz/*.npz
      └─ alt: pkl_to_csv_without_hands.py → mimic/csv_to_npz.py
      ▼
.npz Keys: fps, joint_pos (T,29), joint_vel (T,29), body_pos_w (T,30,3), body_quat_w (T,30,4),
           body_lin_vel_w (T,30,3), body_ang_vel_w (T,30,3)
      ▼
BC/robots/g1_29dof/phase1/motions/*.npz   (die alphabetisch erste Datei wird automatisch genommen)
```

- **Aktueller Clip:** `143_merged_filtered.npz`, 155 Frames bei 50 fps, also **3,1 s**.
- Liegt kein `.npz` in `motions/`, fällt Phase 1 (und damit Unified) mit lauter Warnung auf den
  Gangnam-Platzhalter zurück.
- Qualitätscheck vor dem Training (siehe `phase1/motions/README.md`): vollständiger Zyklus
  Greifen–Fangen–Zurückziehen, Handgelenkshöhe beim Fangen passend zu `catch_height_range`, keine rutschenden Füße.
- **Bekannte Datenprobleme** laut Protokollen: Zittern und umknickende Handgelenke, Selbstkollisionen,
  Ballposition relativ zum Roboter verschoben oder verzögert, Ball nur in wenigen Aufnahmen brauchbar.
  Die Ball-Trajektorie aus dem Mocap wird **im Training nicht verwendet**, der Ball ist dort rein synthetisch.

## 6. Gemeinsame Task-Basis (alle drei Ball-Tasks)

| Parameter | Wert |
|---|---|
| Roboter | `UNITREE_G1_29DOF_MIMIC_CFG` (29 DoF, keine Hände) |
| Physik / Control | `sim.dt = 0.005`, `decimation = 4`, also **50 Hz Policy** |
| Envs | 4096, Abstand 2,5 m, flache Ebene |
| Aktionen | `JointPositionAction`, alle Gelenke, `scale = UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` (pro Gelenk 0,25·effort/stiffness), `use_default_offset=True` |
| Ball | `RigidObject` Kugel r = 0,05 m, 0,15 kg, Reibung 0,5, Restitution 0,6. DR beim Start: Reibung 0,3–0,8, Restitution 0,4–0,8, Masse ±0,05 kg |
| „Hände“ | `HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]` |
| Kontaktsensor | `Robot/.*` (ganzer Körper, wird von `undesired_contacts` gebraucht) |
| Observation-History | 5 Frames (Policy und Critic) |
| PPO | Actor/Critic `[512,256,128]` ELU, lr 1e-3 adaptiv, γ 0,99, λ 0,95, 24 Schritte/Env/Iteration, 30k Iterationen max., Speichern alle 500, `empirical_normalization=False` |

### 6.1 Observation-Layout (Vertrag zwischen den Tasks)

Alle drei Tasks haben **dieselbe Slot-Reihenfolge und dieselben Dimensionen**. Wo einem Task ein
Wert fehlt, steht `dummy_zeros`. Dadurch sind Checkpoints zwischen den Tasks shape-kompatibel.

**Policy** (pro Frame, ×5 History):

| # | Slot | Dim | Phase 1 | Phase 2 | Unified |
|---|---|---|---|---|---|
| 1 | `motion_command` (Referenz joint_pos+vel) | 58 | echt | 0 | echt |
| 2 | `ball_state` = `ball_pos_vel_b` | 6 | 0 | echt | echt |
| 3 | `ball_relative` = `ball_to_hands_b` | 3 | 0 | echt | echt |
| 4 | `hand_pos` = `hand_pos_b` | 6 | 0 | echt | echt |
| 5 | `ball_intercept` = `ball_intercept_b` | 3 | 0 | echt | echt |
| 6 | `motion_anchor_ori_b` | 6 | echt | 0 | echt |
| 7 | `base_ang_vel` (×0,2, Rauschen ±0,2) | 3 | echt | echt | echt |
| 8 | `projected_gravity` | 3 | echt | echt | echt |
| 9 | `joint_pos_rel` | 29 | echt | echt | echt |
| 10 | `joint_vel_rel` (×0,05, Rauschen ±1,5) | 29 | echt | echt | echt |
| 11 | `last_action` | 29 | echt | echt | echt |

Das ergibt nach eigener Rechnung **175 Dimensionen pro Frame**. `context/merge_changes.md` nennt 174;
die tatsächliche Größe beim Laden prüfen (`Actor Model: ... input_dim`).

**Critic** (sauber, ohne Rauschen): `command`(58) · `ball_state`(6) · `ball_relative`(3) · `hand_pos`(6) ·
`ball_intercept`(3) · `motion_anchor_pos_b`(3) · `motion_anchor_ori_b`(6) · `body_pos`(42) · `body_ori`(84) ·
`base_lin_vel`(3) · `base_ang_vel`(3) · `projected_gravity`(3) · `joint_pos`(29) · `joint_vel`(29) ·
`actions`(29). Das sind **307 pro Frame**. Die Tasks füllen die Slots nach demselben Muster wie bei der Policy.

**Konventionen:** Alle Ball- und Hand-Observations liegen im **Yaw-Frame des Roboters** und sind
relativ zur Root. Sie enthalten keine Weltkoordinaten (Env-Origins würden sonst durchsickern). Das
ist derselbe Frame, der auf der Hardware aus VICON berechnet werden soll.
`BallCommand.command` (Weltkoordinaten) ist **nur für Logging** da und darf nicht als Observation dienen.

## 7. Task-Spezifikationen

### 7.1 Phase 1: `Unitree-G1-29dof-BallCatch-Phase1` (Motion Tracking)

- BeyondMimic-Stil: `MotionCommand` mit Anchor `torso_link`, 14 getrackten Bodies, adaptivem Sampling
  über Fehler-Bins, Teleport-Reset am Clip-Ende, Episodenlänge 30 s.
- Rewards: Tracking-Exp-Kernel (Anchor pos/ori je 0,5; Body pos/ori/lin_vel/ang_vel je 1,0) plus
  Regularisierung (joint_acc, joint_torque, action_rate −0,1, joint_limit −10, undesired_contacts −0,1
  ohne Knöchel und Handgelenke).
- Terminierungen: `anchor_pos` (z > 0,25), `anchor_ori` (0,8), `ee_body_pos` (z > 0,25 an Knöcheln und Handgelenken), Timeout.
- DR: Material 0,3–1,6/0,3–1,2/Restitution 0–0,5, Default-Joint-Offset ±0,01, Torso-CoM, Torso-Masse −1…+3 kg, aggressive 6-DoF-Pushes.
- Kein Ball in der Szene, Ball-Slots sind Nullen. Runner: `BasePPORunnerCfg` (std 1,0).

### 7.2 Phase 2: `Unitree-G1-29dof-BallCatch-Phase2` (RL-Fangen)

- Stehen und Fangen ohne Motion-Referenz. Reset: Root zufällig ±0,5 m, **Yaw ±π**, Gelenkgeschwindigkeit zufällig.
- `BallCommand` (ein Wurf pro Episode):
  1. Nach dem Reset wird der Ball am Spawn **geparkt**. Der Spawn liegt im Heading-Frame: Distanz 1,0–2,5 m,
     lateral ±0,75 m, Höhe +0,8–1,5 m über der Root.
  2. Nach `throw_delay_range` 0,5–1,5 s folgt der ballistische Wurf auf die Fangzone. Die Fangzone liegt 0,4 m
     vor dem Roboter, lateral σ 0,05 m, **+0,25–0,45 m über der Root** (Brusthöhe, etwa 1,0–1,2 m).
     Flugzeit 0,3–0,6 s, Geschwindigkeitsrauschen σ 0,3 m/s.
  3. **Gefangen** gilt der Ball, wenn für 25 Schritte (0,5 s) gleichzeitig gilt: Abstand zur nächsten
     Hand < 0,2 m, Relativgeschwindigkeit zu dieser Hand < 0,5 m/s, Ball-z > 0,5 m (Zähler `secured_steps`).
- Rewards:

  | Reward | Gewicht | Bedeutung |
  |---|---|---|
  | `hand_to_ball` | 2,0 | exp(−d²/0,3²) |
  | `catch_success` | 500 | einmalig, effektiv 10 |
  | `ball_secured_hold` | 5,0 | 0,1 pro Schritt |
  | `ball_height_penalty` | −5 | Ball unter 0,5 m |
  | `alive` | 0,15 | |
  | Haltungs-Krücken | | `base_height` −10 @0,76 m, `flat_orientation` −5, `joint_deviation_*`, `feet_slide`, `undesired_contacts` −1 (ohne Knöchel und Handgelenke), weitere Locomotion-Regularisierung |

- Terminierungen: Timeout 20 s, `base_height` < 0,2, `bad_orientation` 0,8, **`ball_caught` (Erfolg)**,
  `ball_dropped` (z < 0,1), `ball_missed` (horizontal > 4 m).
- Curriculum:
  - Wurfverteilung von leicht auf final über Schritte 50k → 250k.
  - Haltungs-Tapers über Schritte 100k → 400k: `base_height` −10 → −2,5, `flat_orientation` −5 → −1, Arm-Deviation −0,1 → −0,02.
  - Play-Cfg setzt `curriculum=None` und spielt damit die finale Verteilung.
- Runner: `Phase2PPORunnerCfg` (**init_noise_std 0,2**, für Fine-Tuning).

### 7.3 Unified: `Unitree-G1-29dof-BallCatch-Unified` (empfohlene Richtung)

- DeepMimic-Stil: Tracking-Rewards aus Phase 1 **und** Ball-Rewards aus Phase 2 gleichzeitig, ohne Dummy-Slots,
  **ohne** Haltungs-Krücken und ohne Roboter-Reset-Events (der `MotionCommand` besitzt die Resets).
- `SyncedBallCommand`: Abwurf bei `launch_time = CATCH_MOTION_TIME ± 0,05 s − flight_time`, damit die
  Ankunft mit dem Fang-Frame im Mocap zusammenfällt.
  - Startet eine Episode nach dem Abwurffenster, gibt es keinen Wurf; dann wird reines Tracking trainiert.
  - Wenn der Clip wrappt oder resampelt wird, wird der Ball neu geparkt und neu scharf geschaltet.
  - **`motion` muss in `CommandsCfg` vor `ball_throw` stehen.**
- Terminierungen: Tracking aus Phase 1 (`ee_body_pos` gelockert auf 0,3) plus Ball-Terminierungen aus Phase 2.
  Drops zählen als Fehler im adaptiven Sampler, der dadurch das Fangfenster häufiger sampelt.
- Curriculum:
  - Wurfverteilung 50k → 250k.
  - Body-Tracking-Tapers (pos/ori 1,0 → 0,4, vel 1,0 → 0,5) über 150k → 500k.
  - Anchor-Tracking bleibt voll gewichtet.
- Pushes 6-DoF alle 2–5 s. Runner: `BasePPORunnerCfg` (std 1,0).
  Optionaler Warmstart: `--resume ... --agent.policy.init_noise_std=0.3`.
- **Vor ernsthaftem Training:** `CATCH_MOTION_TIME` (in `unified/ball_catch_env_cfg.py`) auf den
  Ballkontakt im aktuellen Clip setzen. Der Clip ist 3,1 s lang, 2,0 s ist ein ungeprüfter Platzhalter.
  Außerdem prüfen, ob die Handgelenkshöhe beim Fangen zu `catch_height_range` passt.

## 8. MDP-Bausteine (`BC/mdp/`)

| Datei | Inhalt |
|---|---|
| `commands/motion_command.py` | `MotionLoader`, `MotionCommand(Cfg)`: Referenzzustände, adaptives Bin-Sampling, Metriken `error_*`, `sampling_*` |
| `commands/ball_command.py` | `BallCommand(Cfg)`: Parken, Wurf, `secured_steps`, Metriken `ball_height`, `ball_secured_time`, `hand_ball_distance`. Hilfsmethoden `_sample_spawn`, `_write_parked_state`, `_launch`, `_update_catch_state` |
| `commands/synced_ball_command.py` | `SyncedBallCommand(Cfg)`: Abwurf an die Motion-Phase gekoppelt (nur Unified) |
| `observations.py` | Motion-Obs (`motion_anchor_*`, `robot_body_*`), Ball-Obs (`ball_pos_vel_b`, `ball_to_hands_b`, `ball_intercept_b`, `hand_pos_b`), `dummy_zeros` |
| `rewards.py` | Tracking-Exp-Rewards, `energy`, `hand_to_ball_distance_exp`, `ball_caught_bonus`, `ball_secured`, `ball_height_penalty` |
| `terminations.py` | Tracking-Terminierungen (`bad_anchor_*`, `bad_motion_body_pos*`), `ball_caught`, `ball_below_height`, `ball_far_from_robot` |
| `events.py` | `randomize_joint_default_pos`, `randomize_rigid_body_com` |
| `curriculums.py` | `modify_reward_weight_linear`, `ball_throw_curriculum` (lineare Interpolation über `common_step_counter`) |

`ball_intercept_b`: Schnittpunkt der Ballparabel (absteigend) mit der Ebene 0,35 m über der Root.
Ausgabe `[x_b, y_b, t_go]`, `t_go` auf [0, 3] s geklemmt. Wenn die Ebene nie erreicht wird, ist
`t_go = 0` und die aktuelle Position wird zurückgegeben.

## 9. Invarianten & Stolperfallen

1. **Obs-Vertrag:** Ändert man einen Slot in einem Task (Reihenfolge, Dimension, Skalierung, Rauschen),
   muss man ihn in *allen* Tasks und *beiden* Gruppen spiegeln. Sonst brechen Transfer und Warmstart,
   und alte Checkpoints werden inkompatibel.
2. **Action-Skalierung** ist in allen Tasks `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE`. Eine künftige
   Deploy-Config muss dieselbe verwenden.
3. Isaac Lab multipliziert Rewards mit `weight × dt` (dt = 0,02). Einmal-Boni werden deshalb als `wert / dt` gewichtet.
4. Reihenfolge im Step von Isaac Lab: Terminierungen → Rewards → Reset → Command-Update.
   Schwellwerte der Commands und der Terminierungen dürfen sich nicht widersprechen (dieser Bug ist in P0 behoben).
5. `undesired_contacts` muss Handgelenke (bzw. später Hand-Links) ausschließen, sonst wird das Fangen bestraft.
6. Checkpoints von vor dem 2026-08-10 (Stand vor P0/P1: andere Obs-Skalen, ohne `ball_intercept`) sind **inkompatibel**.
7. Bei Play und Evaluation `curriculum=None` setzen, sonst wird auf der leichten Anfangsverteilung evaluiert.
8. Die Curriculum-Schritte sind auf etwa 30k Iterationen ausgelegt (24 Schritte pro Iteration). Bei anderer Laufzeit skalieren.

## 10. Trainings-Historie & Artefakte

| Run | Task | Motion | Ergebnis |
|---|---|---|---|
| `ballcatch_phase1/2026-07-27_13-32-10` | Phase 1 | Gangnam-Platzhalter | model_99 (Smoke-Test, veraltet, inkompatibel) |
| `ballcatch_phase1/2026-07-27_13-41-06` | Phase 1 | – | leer/abgebrochen |
| `ballcatch_phase1/2026-08-31_14-45-09` | Phase 1 | 143_merged_filtered | model_1 (Testlauf) |
| `ballcatch_phase1/2026-08-31_15-06-41` | Phase 1 | 143_merged_filtered | **model_1500**, 1657 Iterationen, Tracking konvergiert (siehe Kapitel 2) |
| `ballcatch_phase2/` | Phase 2 | – | leer |
| Unified | – | – | nie gestartet |

## 11. Offene Punkte & nächste Schritte (priorisiert)

1. **`CATCH_MOTION_TIME` bestimmen** (per `replay_npz.py` auf `143_merged_filtered.npz`), `catch_height_range`
   gegen die Handgelenkshöhe prüfen, dann **Unified trainieren**, parallel Phase 2 als A/B-Vergleich
   (Warmstart von `2026-08-31_15-06-41/model_1500.pt`).
2. Metrik für die Fangrate etablieren (`Episode_Termination/ball_caught` gegen `ball_dropped`/`ball_missed`).
   Entscheidung, welche Linie (zwei Phasen oder Unified) weiterverfolgt wird.
3. Ballgröße prüfen: Simuliert sind ∅ 10 cm und 0,15 kg. Die Mocap-Gruppe verwendet inzwischen einen
   größeren Ball bzw. einen Fußball. Die Sim-Parameter an den realen Ball anpassen.
4. Die statische Handaufnahme (laut Antrag) als Geometrie modellieren und Fangdetektion bzw. `catch_radius`
   darauf abstimmen. Dex3-Hände sind laut Antrag **nicht** nötig.
5. Sim2Real-Härtung: Rauschen, Verzögerung von 1–2 Schritten und Dropout auf Ball-Observations;
   Aktuatorverzögerung als DR; Wurfverteilung an echte VICON-Würfe fitten.
6. Sim2Sim in MuJoCo, ONNX-Export, Deploy-Config (`deploy/robots/g1_29dof/config/`),
   VICON-Livepipeline (Tracking der Roboterbasis, ballistischer Kalman, Vorhersage über die Latenz,
   **identische** Yaw-Frame-Mathematik wie in `observations.py`).
7. Mehr Clips bzw. besseren Clip auswählen (`pkl_isaac_lab_*`). `MotionCommand` lädt aktuell nur **eine** Datei.

## 12. Kontextdokumente

`context/` (nicht versioniert) enthält:
- `PG-Antrag-H-ReACT.pdf`: offizieller Antrag mit den Minimalzielen
- `Protokolle/`: Wochenprotokolle vom 22.05. bis 10.08.2026 (Teamaufteilung, Stand anderer Subgruppen)
- `g1_catching_optimization_plan.md`: Analyse, Befunde F-1…F-13, Roadmap P0–P3 (Sim2Real, Dex3-Konzept)
- `p0_changes.md`, `p1_changes.md`, `merge_changes.md`: Changelogs der Umsetzung vom 10.08.
- `LATENT_pipeline.md`, `LATENT_mimic_adjustments.md`: Referenz aus dem LATENT-Tennis-Paper
  (Ideen noch nicht vollständig übernommen)
- `Phase2_BallCommand_plan.md`, `debug_instructions.md`: ältere Planungs- und Debug-Stände (teilweise überholt,
  z. B. Multi-Throw-Logik und Debug-Prints wurden entfernt)
- `H-React.drawio(.xml)`: Architekturdiagramm der Gesamtpipeline

Bei Widersprüchen gilt: **Code > dieses Spec-Dokument > Kontextdokumente.**

## 13. Glossar

- **Mimic / Phase 1:** Imitation einer Referenzbewegung per Tracking-Rewards (BeyondMimic-Rezept).
- **Anchor:** Referenz-Body (`torso_link`), relativ zu dem Body-Positionen verglichen werden.
- **Yaw-Frame:** Roboter-Root-Frame nur mit der Gier-Rotation (ohne Roll und Pitch).
- **Secured:** Ball nah an einer Hand, langsam relativ zu ihr, über dem Boden. 25 Schritte in Folge bedeuten „gefangen“.
- **GMR:** General Motion Retargeting, Mensch → G1.

---

## 14. Änderungslog

> Neue Einträge **unten** anhängen. Format:
> `### YYYY-MM-DD: Kurztitel` + Stichpunkte (was, warum, betroffene Dateien bzw. Kapitel).
> Die Einträge vor 2026-09-28 sind aus der Git-Historie rekonstruiert.

### 2026-06-19: Ball-Catching-Task angelegt
- Phase-1-Task als Kopie des Gangnam-Mimic-Tasks (`ca57347`). Pfade für Unitree-Modell/ROS gesetzt, RSL-RL-Kompatibilität.

### 2026-06-25 bis 2026-07-02: Phase 2 aufgebaut
- Ball-Objekt, `BallCommand`, Ball-Observations, Fang-Rewards, Terminierungen, Ball-DR (Masse, Material)
  (`8240bb8` … `2304c13`).

### 2026-07-17: Mocap-Konvertierung
- `scripts/pkl_to_csv_without_hands.py`: GMR-.pkl → CSV, Handspalten verworfen (`93de3a4`).

### 2026-07-26/27: Transfer Phase 1 → Phase 2
- Obs-Padding mit `dummy_zeros`, identische Shapes. Fix in `cli_args.py`: `--experiment_name` wird übernommen,
  damit `--resume` über Tasks hinweg funktioniert (`c5bd979`, `43a1391`).

### 2026-08-10: P0-Fixes (`4cf10f5`)
- Ball- und Hand-Observations im Yaw-Frame, Würfe im Heading-Frame, Zielhöhe Brust, gleiche Action-Skalierung
  in beiden Phasen, vereinheitlichte Obs-Skalen, `undesired_contacts` ohne Handgelenke, Ein-Wurf-Episoden
  mit Wurfverzögerung, neue Fangdetektion (`secured_steps`, Einmal-Bonus, Erfolgs-Terminierung), Hygiene.
  Details: `context/p0_changes.md`.

### 2026-08-10: P1 Content & Transfer-Integrität (`0ff1d94`)
- Auto-Discovery `phase1/motions/`, `ball_intercept_b`, gespiegelte DR, Curriculum (Würfe + Haltungs-Tapers),
  `Phase2PPORunnerCfg` (std 0,2). Details: `context/p1_changes.md`.

### 2026-08-10: Unified-Task (`1e4515f`)
- `Unitree-G1-29dof-BallCatch-Unified` und `SyncedBallCommand`. Details: `context/merge_changes.md`.

### 2026-08-31: Phase 1 auf echten Mocap-Daten trainiert
- Clip `143_merged_filtered.npz` in `phase1/motions/`, Run `2026-08-31_15-06-41` (1657 Iterationen, konvergiert).

### 2026-09-28: Direkte Konvertierung .pkl → .npz, Plan-Dokumente verschoben (`5362bca`)
- `scripts/mimic/pkl_to_npz.py`: GMR-.pkl → .npz in einem Schritt, ganzer Ordner pro Isaac-Session.
- `.gitignore`: `pkl_isaac_lab_*`-Ordner. Plan- und Changelog-Dokumente aus dem Root nach `context/` verschoben.

### 2026-09-28: Spec-Dokument & CLAUDE.md eingeführt
- `unitree_rl_lab_spec.md` neu angelegt (Stand-Aufnahme aller Kapitel).
- `CLAUDE.md` neu: Pflicht, substanzielle Änderungen zuerst in der Spec und dann im Änderungslog festzuhalten.
