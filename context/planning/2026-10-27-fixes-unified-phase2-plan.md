# Implementierungsplan: Fixes 1–4 für Phase 2 und Unified

Plan vom 2026-10-27, **umgesetzt am 2026-10-28** (Abweichungen im nächsten Abschnitt). Grundlage sind die Befunde in `context/analysis/2026-10-27-phase2-final.md` (Belastungstests des Phase-2-Baselines `model_29999.pt`), der Datensatz aus Spec-Abschnitt 5.2 und die Bewertungs-Suite `scripts/rsl_rl/eval_catch.py`. Getestet und trainiert wird auf den Workstations, sobald GPU-Zeit frei ist.

## Umsetzungsstand 2026-10-28

Alle vier Fixes sind auf dem Branch `fixes-2026-10-28` umgesetzt, ein Commit pro Fix (3 → 4 → 2 → 1), jeweils mit Spec und Changelog. Getestet auf Workstation 2 in einer Kopie neben dem laufenden Phase-1-Training (`~/tmp/fixes/unitree_rl_lab`, `PYTHONPATH` auf die Kopie). Was ausdrücklich noch fehlt: echte Trainingsläufe und die Suite auf neu trainierten Checkpoints.

Abweichungen vom Plan unten:

- **Fix 4:** 84 statt 85 gültige Würfe. Die Grenze 0,10 m verwirft außer Clip 51 auch Clip 82. Dessen Zeitversatz ist normal (−0,08 s), aber seine rechte Hand liegt in der zweiten Hälfte 0,11–0,15 m neben der Handfläche des Menschen. Die Annahme „alle sauberen Clips ≤ 0,07 m“ stimmte nicht ganz; alle jetzt gültigen Clips liegen bei ≤ 0,07 m. Der Ablehnungsgrund heißt neutral „arms … off the human of the recording“. Die G1-Konstanten sind schon mit Fix 4 nach `motion_data.py` gewandert (nicht erst mit Fix 1).
- **Fix 2:** Die Begrenzung läuft über das `clip` von Isaac Labs `JointAction`. `export_deploy_cfg` schreibt die Gelenkgrenzen damit nach `deploy.yaml`, und der C++-`JointAction` wendet `clip` schon an. Der Deploy-Punkt aus 4.4 ist damit erledigt; auf dem echten Roboter einmal prüfen.
- **Fix 1, Unified:** Die gehaltene Fangpose dauert 5 s statt 3,5 s (`END_HOLD_S = HOLD_AFTER_CATCH_S + 2`). Mit 3,5 s endet der Clip bei Catch + 3,8 s, die Haltephase aber erst bei Bestätigung (bis ca. 1,5 s nach der Ankunft) + 3 s. Die meisten Fänge würden dann nie abgeschlossen, weil der Roboter vorher auf den nächsten Zyklus zurückgesetzt wird. Zyklen dauern jetzt 12,2–15,5 s (Episode 20 s).
- **Fix 1, Unified:** `SyncedBallCommandCfg.ball_data_dir` ist durch `catch_data` ersetzt. Die Config berechnet die Daten mit `motion_data.catch_data(...)` beim Import, der Command prüft, ob die G1-Konstanten zum Modell passen.
- **Fix 1, Suite:** Das Szenario `hold` behält für Phase 2 die alte Wartezeit 0,5–1,5 s, damit es wie bei der Baseline ca. 9 s Halten misst. Neue Ausgaben: `confirmed_pct`, `after_confirmation_pct` (wie bestätigte Fänge enden; `ball_caught` = Haltephase geschafft), `hold_after_confirm_s_median`, im Report die Spalte `full`.
- **Fix 2, Suite:** `pct_time_target_outside_range` misst gegen die **weichen** Grenzen. Ein auf die harte Grenze begrenzter Sollwert zählt dort weiter als „außerhalb“ (98 % bei den Fuß-Roll-Gelenken). Die Erwartung „0 %“ aus 4.4 gilt für die neue Kennzahl `pct_time_target_beyond_hard_limit`.
- **Alter Phase-2-Checkpoint auf dem neuen Code:** Mit der Begrenzung fällt er in 90 % der Episoden, weil er auf dem Drehmoment am Fuß-Roll-Anschlag stand. Ohne Begrenzung (nur als Diagnose in der Testkopie) fängt er bei 0,5–8 s Wartezeit 87 %. 99,9 % seiner bestätigten Fänge schaffen die Haltephase (genau 3,0 s), im Szenario `hold` halten 98,5 % den Ball 12 s lang. Ein neues Training ist also Pflicht. Die „neue Referenz“ aus Schritt 5 des Ablaufs mit dem alten Checkpoint entfällt, die Vergleichsbasis sind die ersten neu trainierten Läufe.
- **Fix 1, Unified, Wurfanteil:** Gemessen (kinematisches Abspielen, 256 Roboter × 60 s) werfen 57 % der abgeschlossenen Zyklen statt der geplanten ca. 70 %. Rechnerisch starten 65 % der Zyklen vor dem Abwurf; kurze Zyklen ohne Wurf enden im Messfenster häufiger. 0,09 Würfe pro Roboter und Sekunde, Wartezeit bis zum Abwurf im Median 3,2 s, bis 7,1 s. 30-s-Episoden werfen nicht öfter, die Episodenlänge bleibt 20 s.
- **Neu entdeckt, nicht umgesetzt:** Das adaptive Sampling zählt in Unified auch Ballabstürze als Fehlschläge, und zwar im Zeitfenster des Abstürzens, also nach dem Abwurf. Je mehr Fehlschläge Ballabstürze sind, desto öfter starten Zyklen nach dem Abwurf und damit ohne Wurf. Nur die 35 % im Vorlauf werfen sicher. Vorschlag: In Unified nur die Tracking-Abbrüche (`anchor_pos`, `anchor_ori`, `ee_body_pos`) als Fehlschläge zählen (neues Config-Feld im `MotionCommandCfg`). Das ändert das dokumentierte Verhalten, deshalb erst nach Rücksprache.
- **Fix 1, Phase 2, Stillstand-Terme ab 50k Schritten eingeblendet:** Ein Pilotlauf mit den Gewichten −10 / −5 von Anfang an (2048 Envs, 250 Iterationen) hat gelernt, sich nach 7 Schritten fallen zu lassen. Ein zappelnder, untrainierter Roboter bewegt sich schnell, die quadratische Strafe kostet dann mehr als ein Sturz (ca. −17 gegen −6 pro Episode). Mit Gewicht 0 lernt derselbe Pilot zu stehen wie am 2026-10-22 (Episodenlänge 95,6, 32 % Stürze, 69 % erreichen den Wurf). Deshalb blendet das Curriculum beide Terme über die Schritte 50k → 250k ein, zusammen mit den langen Wartezeiten. Im ersten vollen Lauf um Iteration 2000–10 000 prüfen, ob Episodenlänge und `bad_orientation` stabil bleiben, während die Gewichte steigen.
- **Fehlerpfad der Suite getestet:** `eval_catch.py` mit nicht vorhandenem Checkpoint gibt den Traceback aus, schreibt kein JSON und endet nach 10 s. `wait` für Unified läuft mit einem Smoke-Checkpoint durch.

Die Knie bleiben bewusst unverändert (Entscheidung vom 2026-10-27): Die Referenzen liegen fast immer unter der weichen Kniegrenze, beobachtet wird, ob die `joint_limit`-Strafe das Training spürbar stört. Alle Fixes unten sind so gebaut, dass sie diese Entscheidung nicht berühren (Begrenzung auf die **harten** Grenzen, nicht die weichen).

## 0. Überblick

| Fix | Befund (Baseline 2026-10-27) | Phase 2 | Unified | Phase 1 | Aufwand |
|---|---|---|---|---|---|
| 1 Warten und Halten | Wurf nach 6–9 s Warten: 28 % Fänge, 28 % Stürze, 82 cm Wandern, 95° Drehung; nach dem Fang läuft er mit dem Ball 4 m weg; in Unified werfen nur 29 % der Zyklen; 20 Clips enden < 1 s nach dem Fang | lange zufällige Wartezeit, Haltephase nach dem Fang, Stillstand-Terme | Referenz vorn verlängern (Warten), Clip nach dem Fang abschneiden und die Fangpose halten (Halten), Haltephase | unverändert | groß |
| 2 Gelenk-Sollwerte begrenzen | Fuß-Roll-Sollwerte 2–4 rad außerhalb des Gelenkbereichs, Gelenk am Anschlag, Drehmoment beim Warten zu 100 % gesättigt, Füße 14–21° auf der Kante; Schulter presst den Ball mit 236 N | Begrenzung + Überschreitungs-Strafe + Drehmoment-Strafe + flache Füße | Begrenzung + Überschreitungs-Strafe | Begrenzung + Überschreitungs-Strafe | mittel |
| 3 Latenz breiter | 4–6 Schritte Latenz kosten 6 Prozentpunkte, echte Latenz unbekannt | Wahrnehmungsverzögerung 1–5 statt 1–3 Schritte | dasselbe | – | klein |
| 4 Clip 51 | Zeitliche Ausrichtung zur VICON-Aufnahme unsicher, Hände 12 cm neben dem Menschen | – | Gültigkeitsprüfung um den Mensch-Vergleich erweitern (verwirft 51) | – | klein |

Bewusst **nicht** in diesem Paket: Entropie (erst beobachten über `Policy/mean_std`), eigene Bereitschaftspose für Phase 2 (teilweise über die Stillstand-Terme abgedeckt), Stoß-Intervalle (werden durch die längeren Episoden ohnehin häufiger, prüfen), Aktor-Verzögerung, GRU-Policy, Sim2Sim, Endeffektor-Modell, Knie.

## 1. Arbeitsweise

- Ein eigener Branch von `main`, z. B. `fixes-2026-10-28`. `main` bleibt der getestete Stand, bis morgen alles geprüft ist.
- Ein Commit pro Fix, in der Reihenfolge 3 → 4 → 2 → 1 (klein und unabhängig zuerst, der große Umbau zuletzt). Jeder Commit aktualisiert Spec und Changelog (Pflicht laut `CLAUDE.md`) und markiert im Changelog „implementiert, ungetestet“, bis die Tests gelaufen sind.
- Heute nur statische Prüfungen: `python3 -m py_compile` auf dem Laptop (dort gibt es weder Isaac noch numpy). Keine GPU-Läufe.
- Alte Checkpoints laden nach den Fixes zwar weiter (gleiche Netzform), verhalten sich aber anders (begrenzte Sollwerte, andere Episoden). Deshalb alle neuen Läufe **von null**, kein `--resume` aus alten Läufen.

## 2. Fix 3: Latenz breiter (Phase 2 und Unified)

**Änderungen**

- `mdp/commands/ball_command.py`, `BallCommandCfg.obs_delay_steps_range`: Default `(1, 3)` → `(1, 5)`. Der Ringpuffer der Wahrnehmung wird aus diesem Default dimensioniert (`buffer_len = obs_delay_steps_range[1] + 1`), deshalb muss der Default die größte Verzögerung enthalten.
- `robots/g1_29dof/phase2/ball_catch_env_cfg.py` und `unified/ball_catch_env_cfg.py`, `CurriculumCfg.ball_throws.params["final"]["obs_delay_steps_range"]`: `(1, 3)` → `(1, 5)`. Der leichte Startwert `(1, 1)` bleibt.
- Spec 7.2 (Wahrnehmung), 9 (Fallstrick 15 erwähnt die Pufferlänge), Changelog.

**Warum 1–5:** 5 Steuerschritte sind 100 ms. Die echte Kette VICON → MQTT → Roboter ist ungemessen (Spec, offener Punkt 4), 80–120 ms kosteten im Test 6 Prozentpunkte. Sobald die Latenz gemessen ist, wird der Bereich daran angepasst.

**Folge für die Suite:** Das Szenario `delay` (4–6 Schritte) liegt danach teilweise im Trainingsbereich. Es bleibt trotzdem unverändert, damit die Zahlen mit der Baseline vergleichbar sind.

**Test (morgen):** Smoke-Run Phase 2 und Unified. Im Szenario `delay` sollte die Fangquote kaum noch unter `base` liegen.

## 3. Fix 4: Clip 51 über die Gültigkeitsprüfung ausschließen (Unified)

Statt Clip 51 per Namen auszuschließen, wird die Gültigkeit einer Ballaufnahme um den Mensch-Vergleich aus `scripts/mimic/check_dataset.py` erweitert. Das trifft genau die Clips, deren zeitliche Ausrichtung unsicher ist (51 und 52), auch bei künftigen Daten.

**Änderungen**

- `tasks/ball_catching/motion_data.py`:
  - `compare_with_human(clip, rec, alignment)` aus `check_dataset.py` hierher verschieben (nur numpy).
  - Neue Konstante `MAX_ARM_RMSE = 0.10` m: Ellbogen und Hände des Clips dürfen im Mittel höchstens 10 cm von den skalierten menschlichen Punkten abweichen. Gemessen: alle sauberen Clips ≤ 0,07 m, Clip 51 0,12 m, Clip 52 0,19–0,22 m.
  - `compute_catch_info` ruft den Vergleich nach der Ausrichtung auf und setzt `reason = "arms off the human ..."`, wenn die Grenze überschritten ist.
- `scripts/mimic/check_dataset.py`: `compare_with_human` aus `motion_data` importieren statt selbst definieren.
- `scripts/mimic/catch_info.py`: unverändert (zeigt den neuen Grund an).

**Erwartung:** 85 statt 86 gültige Würfe. Abgelehnt werden 52 (Fang außerhalb des Clips), 45 und 101 (Ball zu weit von den Händen) und neu 51 (Arme weichen vom Menschen ab).

**Test (morgen, nur numpy):** `python scripts/mimic/catch_info.py` → „valid catches: 85 of 108“. `python scripts/mimic/check_dataset.py` → unverändert 87 von 89 bei den Armen.

## 4. Fix 2: Gelenk-Sollwerte begrenzen (alle drei Tasks)

### 4.1 Neuer Action-Term

Datei `tasks/ball_catching/mdp/actions.py` (neu), exportiert in `mdp/__init__.py`:

```python
class ClampedJointPositionAction(JointPositionAction):
    """Joint position targets clamped to the asset's joint limits.

    The policy's targets are otherwise unbounded (BeyondMimic action scale): the Phase 2 baseline drove the
    ankle-roll targets 2-4 rad past the joint range and stood on the hard stops with saturated torque. Clamping to
    the hard limits removes that lever without changing the reachable range. ``target_excess`` (sum over the joints
    of |unclamped - clamped|, rad) feeds a penalty, because beyond the clamp the reward has no gradient.
    """

    cfg: ClampedJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]  # hard limits of the asset
        self._target_lo, self._target_hi = limits[..., 0].clone(), limits[..., 1].clone()
        self.target_excess = torch.zeros(self.num_envs, device=self.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        clamped = torch.clamp(self._processed_actions, self._target_lo, self._target_hi)
        self.target_excess = (self._processed_actions - clamped).abs().sum(dim=-1)
        self._processed_actions = clamped


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction
```

- **Harte statt weiche Grenzen:** Die weichen Grenzen (90 % des Bereichs) würden gestreckte Knie verbieten (weiche Untergrenze 0,061 rad) und damit die Knie-Entscheidung vorwegnehmen. Mit den harten Grenzen ist der erreichbare Bereich derselbe wie heute. Nur das Drücken über den Anschlag hinaus fällt weg: Steht der Sollwert auf dem Anschlag und das Gelenk ebenfalls, ist der PD-Fehler null, also gibt es kein Dauerdrehmoment mehr.
- Ein `clip_actions` im Runner wäre keine Alternative: Der BeyondMimic-Action-Scale ist pro Gelenk sehr verschieden (Handgelenk-Yaw ca. 0,09 rad pro Einheit, Fuß ca. 0,44), ein gemeinsames Rohwert-Limit würde kleine Gelenke im Bereich beschneiden.
- In allen drei Configs `ActionsCfg.JointPositionAction = mdp.JointPositionActionCfg(...)` → `mdp.ClampedJointPositionActionCfg(...)`, gleiche Parameter. Der Name des Terms bleibt `JointPositionAction`, damit Logs und `export_deploy_cfg` gleich bleiben. Weil die neue Klasse eine Unterklasse ist, sollte der Deploy-Export sie wie bisher erkennen. Das beim ersten Smoke-Run an `params/deploy.yaml` prüfen.

### 4.2 Strafe für Überschreitung (alle drei Tasks)

- `mdp/rewards.py`: `joint_target_excess(env, action_name="JointPositionAction")` gibt `term.target_excess` zurück.
- Gewicht **−1,0** in allen drei Tasks. Größenordnung: Die Baseline hätte mit ca. 6 rad Überschreitung (beide Fuß-Roll-Gelenke) −1 × 6 × 0,02 = −0,12 pro Schritt gezahlt, mehr als alle Ball-Rewards zusammen. Zu Trainingsbeginn (Rauschen 1,0) fallen nur kleine Überschreitungen an (ca. 0,2 rad, also −0,004 pro Schritt).
- Grund: Hinter der Begrenzung ändert ein noch größerer Rohwert nichts mehr, die Policy merkt es also nicht. Ohne Strafe können die Rohwerte weglaufen. Das stört über die Beobachtung `last_action` (Rohwerte, nicht normalisiert) das Netz.

### 4.3 Nur Phase 2: Drehmoment-Strafe und flache Füße

- `joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)`. Genau wie Phase 1 und Unified, die den Term schon haben. Phase 2 hat bisher nur `energy`, und das ignoriert statisches Drehmoment.
- Neuer Term `feet_flat` in `mdp/rewards.py`: Σ über beide Füße von |projizierte Schwerkraft xy im Fußrahmen|², nur bei festem Bodenkontakt (Kontaktkraft > 50 N, Sensor `contact_forces`, Körper `.*_ankle_roll_link`). Bei 15° Neigung ergibt das sin² ≈ 0,067 pro Fuß. Startgewicht **−2,0**. Ziel in der Suite: mittlere Fußneigung < 5°.
- Unified bekommt `feet_flat` vorerst nicht: Die Füße gehören zu den 14 verfolgten Körpern, ihre Ausrichtung wird also schon belohnt. Nachrüsten, falls die Suite nach dem Unified-Training Fußneigung zeigt.

### 4.4 Folgen

- **Deployment:** Der C++-Deploy muss dieselbe Begrenzung auf die Gelenkgrenzen anwenden, bevor die Sollwerte an den Roboter gehen. In Spec, offener Punkt 9 (Sim2Sim/Deploy) vermerken.
- **Suite:** `joints_at_limit[*].pct_time_target_outside_range` muss nach dem Fix 0 % sein. Erwartet wird weniger Drehmoment-Sättigung an den Fußgelenken und weniger Fußneigung.

**Test (morgen):**
- Smoke-Run aller drei Tasks.
- Suite `base` mit dem alten Phase-2-Checkpoint auf dem neuen Code: Sollwerte außerhalb 0 %. Das Verhalten ändert sich, weil die alte Policy auf das Drücken gegen den Anschlag eingespielt ist. Das ist erwartet und dient nur als Funktionsnachweis.

## 5. Fix 1: Warten und Halten

### 5.1 Gemeinsame Bausteine in `BallCommand` (Phase 2 und Unified)

**Haltephase nach dem bestätigten Fang:**

- Neues Feld `BallCommandCfg.hold_after_catch_s: float = 0.0`. Phase 2 und Unified setzen **3,0 s**.
- Neuer Zustand im Command:
  - `confirmed` (bool): Fang bestätigt, sobald `secured_steps` zum ersten Mal `secure_steps` erreicht.
  - `confirmed_now` (bool): nur im Schritt der Bestätigung wahr.
  - `time_since_confirm`: läuft ab der Bestätigung.
  - `catch_complete`: `confirmed` und `time_since_confirm ≥ hold_after_catch_s`.
- Zurückgesetzt wird beim neuen Wurf (`_reset_catch_state`).
- `mdp/terminations.py`, `ball_caught`: gibt `cmd.catch_complete` zurück. Bei `hold_after_catch_s = 0` ist das wie bisher die Bestätigung selbst. Der Term-Name bleibt (Logging-Kontinuität), er bleibt eine **Truncation** (`time_out=True`).
- `mdp/rewards.py`, `ball_caught_bonus`: `cmd.confirmed_now` statt `secured_steps == secure_steps`. Damit gibt es den Bonus genau einmal pro Wurf, auch wenn die Halte-Serie während der Haltephase reißt und neu beginnt.
- `ball_secured` (Halte-Reward pro Schritt) bleibt unverändert und wirkt in der Haltephase weiter.
- Fällt der Ball in der Haltephase, endet die Episode wie bisher über `ball_dropped`, also als Fehlschlag.
- **Kein neuer Anreiz zum Hinauszögern:** Bestätigung und Zeitlimit enden beide als Truncation mit Bootstrapping. Ob die Policy bestätigt oder nicht, kostet also keinen zukünftigen Halte-Reward, und der Bonus spricht fürs Bestätigen (Logik wie im Fix vom 2026-10-22).
- Neue Metrik `hold_time_after_confirm`.

**Wartezeit-Verteilung (für Phase 2):**

- Neues Feld `BallCommandCfg.throw_delay_log_uniform: bool = False`. Bei `True` wird `time_until_throw` log-gleichverteilt aus `throw_delay_range` gezogen.
- Begründung: Bei 0,5–8 s liegt der Median dann bei 2 s und nur ca. 25 % der Würfe kommen nach mehr als 4 s. Lange Wartezeiten werden also geübt, ohne die Zahl der Würfe pro Trainingsschritt zu halbieren.

### 5.2 Phase 2

- `CommandsCfg.ball_throw`: `throw_delay_log_uniform=True`, `hold_after_catch_s=3.0`.
- `CurriculumCfg.ball_throws`: `"easy": {"throw_delay_range": (0.5, 1.5)}`, `"final": {"throw_delay_range": (0.5, 8.0)}`. Der Curriculum-Interpolator kann Tupel aus Floats.
- **Stillstand-Terme** (neu in `mdp/rewards.py`), aktiv nur **vor dem Abwurf** (`not cmd.launched`) und **in der Haltephase** (`cmd.confirmed`):
  - `stand_still_root`: |v_xy der Basis|² + 0,25·ω_z². Startgewicht **−10**. So kostet das Wandern der Baseline beim Warten (0,16 m/s, 15°/s) ungefähr so viel, wie `alive` einbringt.
  - `stand_still_feet`: Summe der horizontalen Fußgeschwindigkeit bei festem Kontakt (> 50 N). Startgewicht **−5** (Baseline beim Warten ca. 0,08 m/s pro Fuß).
- Episodenlänge bleibt 20 s (Maximum: 8 s Warten + ca. 1 s Flug + 1 s Bestätigen + 3 s Halten = 13 s).
- `push_robot` (alle 4–8 s) bleibt. Durch die längeren Episoden feuert er jetzt tatsächlich, im Smoke-Run prüfen.

### 5.3 Unified: Referenz verlängern und Fangpose halten

**Neue Felder in `MotionCommandCfg`:**

| Feld | Bedeutung | Unified | Phase 1 |
|---|---|---|---|
| `start_hold_s` | Erstes Bild wird so lange davor gesetzt (statisch, Geschwindigkeiten 0) | 6,0 | 0 |
| `start_hold_prob` | Anteil der Zyklen, die gezielt **im** Vorlauf starten (gleichverteilt). Der Rest nutzt das adaptive Sampling über den ganzen verlängerten Clip. | 0,35 | 0 |
| `start_hold_from_begin` | nur für die Suite: alle Zyklen am Anfang des Vorlaufs starten (maximales Warten) | False | False |
| `hold_from_s` | pro Clip: ab dieser Zeit wird der Clip abgeschnitten und dieses Bild gehalten (`None` = Clip vollständig) | Fang + 0,3 s für gültige Fänge | `None` |
| `end_hold_s` | so lange wird das letzte (bzw. abgeschnittene) Bild gehalten | 3,5 | 0 |

**Umsetzung im `MotionLoader`:**
- Pro Clip: Bilder `[0] * n_start + clip[: cut] + [cut − 1] * n_end`.
- In den verlängerten Teilen: `joint_vel`, `body_lin_vel_w` und `body_ang_vel_w` auf 0; Positionen und Ausrichtungen wiederholt.
- `time_step_total` ist die neue Länge.
- Der Loader merkt sich `start_hold_frames`, damit der `SyncedBallCommand` Fang- und Abwurfzeit um den Vorlauf verschieben kann.

**Umsetzung im `MotionCommand._adaptive_sampling`:**
- Mit Wahrscheinlichkeit `start_hold_prob` wird die Startzeit gleichverteilt in `[0, start_hold_frames)` gezogen, sonst wie bisher adaptiv.
- Grund: Im statischen Vorlauf scheitert fast nie etwas. Das adaptive Sampling würde ihn deshalb über die Zeit immer seltener ziehen, und lange Wartezeiten würden aus dem Training verschwinden.

**Warum nicht einfach das letzte Bild halten:** Datenprüfung vom 2026-10-27. In **33 von 86** Clips sind die Hände am Clip-Ende mehr als 15 cm von der Fangpose weg oder weit gespreizt (Abstand bis 0,53 m), die Person legt den Ball schon ab. Eine Referenz, die den Ball loslässt, verträgt sich nicht mit der Haltephase. Deshalb: abschneiden bei Fang + 0,3 s (Hände haben sich gesetzt, mittlere Auslenkung nach dem Fang 9 cm) und diese Pose halten.

**Fangdaten zur Config-Zeit statt zur Laufzeit:**
- `motion_data.py` bekommt die Konstanten `G1_JOINT_NAMES` und `G1_HAND_BODY_INDEXES` (bisher in `catch_info.py`, durch `check_npz_fk.py` gegen das Modell bestätigt).
- Die Unified-Config ruft `catch_infos(...)` einmal beim Import auf (ca. 1–2 s mit numpy). Daraus entstehen:
  - `hold_from_s` für den `MotionCommandCfg`
  - neues Feld `SyncedBallCommandCfg.catch_data` (pro Clip `valid`, `catch_time`, `flight_time`, `catch_point`)
- Der `SyncedBallCommand` rechnet dann nichts mehr selbst aus. Er verschiebt nur `catch_time` um `start_hold_s`.
- `catch_info.py`, `check_dataset.py` und `check_npz_fk.py` importieren die Konstanten aus `motion_data`.

**`SyncedBallCommandCfg`:** `hold_after_catch_s=3.0`. Die Wartezeit-Felder bleiben ungenutzt, der Wurf hängt an der Clip-Zeit.

**Erwartungen:**
- Wurf-Anteil der Zyklen ca. 70 % statt 29 % (0,35 aus dem gezielten Vorlauf + 0,65 × ca. 59 % der gleichverteilten Starts vor dem Abwurf).
- Wartezeiten bis ca. 6,9 s (6 s Vorlauf + 0,86 s Abwurf im Median).
- Jeder gültige Clip erlaubt Bestätigung plus 3 s Halten (Fangpose 3,5 s gehalten).
- Ein verlängerter Clip dauert ca. 11–12 s, die Episodenlänge (20 s) reicht.
- Der Vorlauf trainiert ruhiges Stehen über das Tracking einer statischen Pose. Für Unified sind deshalb vorerst keine eigenen Stillstand-Terme nötig.

**Prüfen beim Implementieren:** Bei Clips mit Abwurf bei ca. 0 s (Clip beginnt erst mit dem Wurf) ist Bild 0 schon Teil der Fangbewegung. Der Vorlauf hält dann eine nicht ganz ruhige Pose. Hinnehmen oder diese Clips später aussortieren, das zeigt der Replay.

### 5.4 Bewertungs-Suite

- `scripts/rsl_rl/eval_catch.py`, Szenario `wait` für Unified: statt abzulehnen `env_cfg.commands.motion.start_hold_from_begin = True`. Wartezeit dann ca. 6,9 s, vergleichbar mit `wait` in Phase 2 (6–9 s).
- Szenario `lateral` bleibt für Unified abgelehnt (das Ziel kommt aus der Aufnahme).
- Szenario `hold` (kein Abbruch beim Fang, 12-s-Episoden) bleibt für beide Tasks und misst jetzt Halten über die 3-s-Haltephase hinaus.
- `eval_catch_all.sh`: Unified-Liste um `wait` ergänzen.
- Neue Ausgaben: `hold_time_after_confirm` und der Anteil der Würfe, die die volle Haltephase schaffen.

### 5.5 Tests (morgen)

- **Smoke-Runs** Phase 2 und Unified mit 64 Envs, 3 Iterationen.
- **Unified** mit `unified_check.py --kinematic` (liegt in `analysis/2026-10-27_imitation_data/`):
  - Wurf-Anteil der Zyklen ca. 70 %
  - Ball weiterhin ca. 0,23 m von den Referenz-Handgelenken
  - Wartezeiten bis ca. 7 s
- **Phase 2**, kurzer Play-Lauf mit Zufalls-Policy oder dem alten Checkpoint:
  - Verteilung der Wartezeiten log-gleich (Median ca. 2 s)
  - Bonus genau einmal pro Wurf (`Episode_Reward/catch_success` ≈ 10 × Fangquote)
  - Episoden enden ca. 3 s nach der Bestätigung

## 6. Doku pro Fix

| Fix | Spec-Abschnitte | Sonstiges |
|---|---|---|
| 3 | 7.2, 9 | – |
| 4 | 5.2 (Gültigkeit, 85 statt 86) | `check_dataset.py` |
| 2 | 3 (neue Datei `mdp/actions.py`), 6 (Action-Term), 7.1–7.3 (Rewards), 8 (MDP-Bausteine), 9 (Fallstrick: unbegrenzte Aktionen), 11 (offener Punkt 9 Deploy) | `REVIEW.md` unberührt |
| 1 | 7.2, 7.3 (Episodenablauf, Vorlauf, Haltephase), 8, 9 (Fallstrick: adaptives Sampling meidet statische Vorläufe), 11 | Suite-Doku im Skriptkopf |

Jeweils Changelog-Eintrag „implementiert 2026-10-27, Tests ausstehend“, nach den Tests um die Ergebnisse ergänzt.

## 7. Ablauf morgen

1. **Branch auf die Workstations holen.** Wie bisher auf dem Laptop: Den Branch `workstation` (Commit für `UNITREE_MODEL_DIR`) auf `origin/fixes-2026-10-28` rebasen und mit `--force-with-lease` pushen. Auf den Workstations dann nur `cd ~/unitree_rl_lab && git pull --rebase`. Nach dem Merge der Fixes in `main` dasselbe noch einmal auf `origin/main`.
2. **Statische Prüfungen (nur numpy):** `catch_info.py` (85 von 108), `check_dataset.py` (87 von 89).
3. **Smoke-Runs:** Phase 1, Phase 2, Unified mit `--num_envs 64 --max_iterations 3 --experiment_name $HOME/tmp/smoke_<task>`.
4. **Funktionsprüfungen:** `unified_check.py --kinematic` (5.5), Suite `base` mit dem alten Phase-2-Checkpoint (4.4), kurzer Phase-2-Play-Lauf (5.5). Dazu die am 2026-10-27 noch ungetesteten Fehlerpfade der Suite:
   - `eval_catch.py` mit nicht vorhandenem Checkpoint: Traceback, keine `.json`, Prozess endet statt zu hängen (Umgebung wird jetzt im `finally` geschlossen)
   - `--task Unitree-G1-29dof-BallCatch-Unified --scenario wait`: sofortige Ablehnung ohne Isaac-Start. Das gilt nur bis Fix 1, danach ist `wait` für Unified erlaubt.
5. **Neue Referenz:** Die Suite mit dem alten Checkpoint `model_29999.pt` auf dem neuen Code komplett laufen lassen (`eval_catch_all.sh`). Die neuen Szenario-Definitionen (Haltephase, Wartezeiten) ändern `base`. Neue Läufe werden gegen diese Tabelle verglichen, nicht gegen die vom 2026-10-27.
6. **Training**, je nach freier GPU-Zeit, jeweils in tmux, ca. 15 000–20 000 Iterationen:
   - Phase 1 als Prüfung, ob die Clips nachmachbar sind
   - Unified (Stufe A)
   - optional Phase 2 von null als Vergleich
7. **Auswerten** mit `eval_catch_all.sh`, Vergleich über `eval_catch_report.py`.
8. **Merge** des Fix-Branchs in `main`, Workstation-Branch rebasen, Changelog um die Testergebnisse ergänzen.

## 8. Risiken und offene Entscheidungen

- **Gewichte sind Startwerte:** Überschreitung −1, flache Füße −2, Stillstand −10 / −5, Drehmoment −1e-5. In den ersten 100 Iterationen auf die `Episode_Reward/*`-Größen achten. Kein Stillstand-Term sollte größer als die Ball-Rewards werden, sonst lernt die Policy, den Ball zu ignorieren.
- **Längere Episoden** (Warten, Halten) bedeuten weniger Würfe pro Iteration, das Lernen des Fangens wird langsamer. Das ist eine Gegenrechnung zu den 29 % → 70 % in Unified. Gegebenenfalls `start_hold_prob` oder die Wartezeit-Obergrenze anpassen.
- **Haltephase und Curriculum:** Die Bestätigung braucht anfangs 25 Schritte, am Ende 50. Die Haltephase von 3 s kommt jeweils dazu.
- **Unterschiedliche Haltereferenz:** Phase 2 hat keine Referenz für das Halten (Stillstand-Terme), Unified hält die Fangpose. In Stufe B (Feinschliff in Phase 2) sind die Stillstand-Terme deshalb das Bindeglied.
- **Abschneiden nach dem Fang** ändert, was Unified nachahmt: kein Ablegen des Balls mehr. Das ist gewollt, weil der Roboter halten soll, aber im Bericht erwähnen.
- **Begrenzung und Deploy:** Am echten Roboter muss dieselbe Begrenzung laufen (4.4).
- **Knie:** unverändert. `Episode_Reward/joint_limit` beobachten, ob er in Phase 1/Unified dauerhaft groß ist.

## 9. Referenzwerte der Baseline (2026-10-27, alter Code)

| Szenario | Gefangen | Fallen gelassen | Gestürzt |
|---|---|---|---|
| base | 99,6 % | 0,4 % | 0,03 % |
| wait (6–9 s) | 27,9 % | 43,9 % | 28,3 % |
| push | 93,6 % | 6,3 % | 0,2 % |
| delay (4–6) | 93,6 % | 6,3 % | 0,07 % |
| lateral (±0,3 m) | 99,5 % | 0,5 % | 0,06 % |
| hold (12 s) | 99,4 % halten bis zum Ende | 0,1 % | 0,6 % |

Fußneigung 21° / 14°, Fuß-Roll-Sollwerte zu 98–99 % außerhalb des Gelenkbereichs, beim Warten Drehung um 15° pro Sekunde, Ball mit 236 N an die linke Brust gepresst. Ausführlich in `context/analysis/2026-10-27-phase2-final.md`.
