# Phase 2: Befunde und gewichtete Vorschläge (Stand 2026-10-27)

> Kopie aus `analysis/2026-10-27_phase2_final/vorschlaege.md` auf Marius' Laptop (außerhalb des Repos). Dort liegen auch die Artefakte, auf die sich die relativen Pfade unten beziehen: Videos, `stress_tests/*.json`, `eval/eval_catch.py`, Bildraster. Checkpoint `model_29999.pt` und TensorBoard-Logs liegen auf Workstation 2 unter `~/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_ballcatch_phase2/2026-10-23_03-35-20/`.

Grundlage: finaler Checkpoint `model_29999.pt` (Lauf `2026-10-23_03-35-20`, Code `29236d9`), deterministische Policy, Play-Config (finale Wurf-Verteilung). Gemessen mit `eval/eval_catch.py` auf Workstation 1 (512 Roboter × 60 s pro Szenario, je 2.500–9.700 Würfe), dazu die Nahaufnahmen und drei Ganzkörper-Videos. Rohdaten: `stress_tests/*.json`, Videos: `videos/`, Bildraster: `frames/sheets/`.

## 1. Kurzfassung

- Die Fangquote von 99,6 % täuscht. Sie gilt nur für genau die Situation, die das Training abdeckt: 0,5–1,5 s warten, Wurf, 1 s halten, Ende. Sobald der Roboter **länger warten** muss (wie am echten Roboter immer), fängt er nur noch 28 % und **stürzt in 28 %** der Fälle.
- Die Hauptursache für das „Rutschen“ ist ein **Exploit der Gelenkanschläge**: Die Policy gibt für die Fuß-Roll-Gelenke Soll-Winkel 2–4 rad außerhalb des Gelenkbereichs vor. Die Gelenke stehen fast immer am Anschlag, mit vollem Drehmoment (beim Warten 100 % der Zeit), und die Füße stehen 14–21° gekippt auf der Kante. Aktionen werden nirgends begrenzt, und Phase 2 bestraft statisches Drehmoment nicht.
- Echtes Gleiten der Füße ist selten (< 1 % der Kontakt-Schritte). Was wie Rutschen aussieht, ist Drehen auf der Fußkante, Trippeln und ständiges Wandern: 15° Drehung pro Sekunde beim Warten, 33 cm Weg pro Fang.
- Gefangen wird **mit den Hand-Links**, aber der Ball wird links an die Brust gepresst (rechter Arm greift quer herüber, mittlere Kontaktkraft am Torso 236 N, rechte Schulter und rechtes Handgelenk fast dauerhaft am Drehmoment-Limit). Ein Fang vorn mit beiden Händen ist das nicht.
- Halten selbst ist stabil (99 % halten den Ball auch 12 s lang), aber der Roboter **läuft dabei mit dem Ball davon** (4 m in ca. 10 s).

## 2. Befunde im Detail

### 2.1 Belastungstests (Fangquote unter Bedingungen außerhalb des Trainings)

| Szenario | Gefangen | Fallen gelassen | Gestürzt |
|---|---|---|---|
| Basis (Play-Config) | 99,6 % | 0,4 % | 0,03 % |
| Reibung realistisch (statisch 1,0 / dynamisch 0,8) | 99,8 % | 0,2 % | 0,01 % |
| Reibung niedrig (0,4 / 0,3) | 99,7 % | 0,3 % | 0,1 % |
| **Langes Warten** (Wurf nach 6–9 s statt 0,5–1,5 s) | **27,9 %** | 43,9 % | **28,3 %** |
| Langes Halten (kein Abbruch beim Fang, 12 s Episode) | 99,4 % halten bis zum Ende | 0,1 % | 0,6 % |
| Stöße ±0,5 m/s alle 1–2 s | 93,6 % | 6,3 % | 0,2 % |
| Latenz 4–6 statt 1–3 Steuerschritte (80–120 ms) | 93,6 % | 6,3 % | 0,07 % |
| Würfe seitlich ±0,3 m statt ±0,1 m | 99,5 % | 0,5 % | 0,06 % |

Beim langen Warten wandert der Roboter bis zum Abwurf im Median 82 cm weg und dreht sich um 95°. Der Wurf ist aber beim Reset relativ zu seiner damaligen Blickrichtung festgelegt und kommt dann von der Seite oder von hinten. Auch im normalen Training dreht er sich beim Warten schon um ca. 15° pro Sekunde, das fällt dort nur wegen der kurzen Wartezeit nicht auf.

### 2.2 Füße und Stand: Gelenkanschlag-Exploit

| Messgröße (Basis) | links | rechts |
|---|---|---|
| Soll-Winkel Fuß-Roll außerhalb des Gelenkbereichs | 98 % der Zeit | 99 % der Zeit |
| mittlere Abweichung Soll zu Ist (Fuß-Roll) | 2,3 rad | 4,0 rad |
| Gelenk innerhalb 5 % am Anschlag | 95 % | 98 % |
| Drehmoment > 80 % des Limits (50 Nm), beim Warten | 100 % | 100 % |
| Fußneigung bei festem Bodenkontakt (Mittel) | 21° | 14° |

- Reines Gleiten (Fuß bewegt sich > 5 cm/s, ohne sich zu drehen) kommt nur in 0,5–0,7 % der Kontakt-Schritte vor. Die Füße drehen sich pro Standphase im Median um 7,6° (90 %-Quantil 20°), der Fußpunkt wandert dabei 4,5 cm (Warten) bzw. 2,1 cm (Fangphase).
- Füße heben ab: beim Warten 1,9-mal pro Sekunde, in der Fangphase 7,7-mal, beim langen Halten 12,6-mal pro Sekunde und Roboter (Trippeln).
- Reibung spielt praktisch keine Rolle: Bei realistischer und bei niedriger Reibung ist das Verhalten fast gleich. Das ist also gelerntes Verhalten, kein Effekt der Reibungs-Randomisierung.
- Korrektur zur ersten Messung (README): Die dort genannten 0,12 m/s „Rutschen“ vor dem Abwurf enthielten Drehen und Abrollen der Füße. Die Feinmessung trennt das. Eine stärkere Strafe für Fußverschiebung allein (auch Idee 5 in `MotionBricks_Relevanz.md`) behandelt deshalb nur das Symptom. Die Ursache sind die Gelenkanschläge.
- Ursache im Code: Die Aktionen sind in allen drei Tasks unbegrenzt (`clip_actions=None`, kein Clipping im `JointPositionAction`). Phase 2 hat nur `energy` (Leistung = Moment × Geschwindigkeit). Statisches Drücken gegen einen Anschlag kostet daher nichts. `dof_pos_limits` greift nur am weichen Limit und ist schwach. Phase 1 und Unified haben wenigstens `joint_torques_l2` (−1e-5).

### 2.3 Haltung

| | Warten | Ball gehalten |
|---|---|---|
| Beckenhöhe (Ziel 0,76 m) | 0,68 m | 0,75 m |
| Kniewinkel (Grundhaltung 38°) | 72° | 40° |
| Becken nach vorn geneigt | 8° | 34° |
| Schulter-Roll (Arme abgespreizt, Grundhaltung 11°) | 32° | 34° |
| Beckengeschwindigkeit | 0,16 m/s | 0,27 m/s |

Er wartet also in tiefer Kniebeuge mit abgespreizten Armen, oft asymmetrisch mit einem seitlich ausgestreckten Arm. Beim Halten beugt er sich stark über den Ball.

### 2.4 Fangstil

- Ball relativ zum Becken (Blickrichtung): 0,20 m vor, 0,14 m links, 0,11 m über dem Becken, also an der linken Brust oder Flanke.
- Linke Hand außen (0,21 m links), rechte Hand mittig (0,02 m): Der rechte Arm greift quer herüber.
- Ballkontakt: beide Hand-Links (`wrist_yaw_link`) praktisch immer, Ellbogen nie. Torso-Kontakt in 100 % der Haltezeit, im Mittel 236 N.
- Beim Halten über 80 % des Drehmoment-Limits: rechte Schulter-Yaw 99,8 %, rechtes Handgelenk-Pitch 89 %, rechte Schulter-Pitch 80 %, rechte Schulter-Roll 62 % der Zeit. Die Soll-Winkel der rechten Schulter liegen 1,3–2,4 rad „im Körper“.
- Im Video: zuerst eine Hand, die den Ball abfängt oder lenkt, dann Einklemmen an der Brust wie bei einem Rugbyball.

### 2.5 Training

- Die Explorations-Streuung `mean_std` ist im Lauf **gestiegen**, von 0,69 (Iteration 9.000) auf ca. 0,93 (ab 22.000). Der Entropie-Bonus (0,005) überwiegt.
- Die Fangquote im Training (stochastisch) lag ab Iteration ca. 15.000 bei 92–95 %, danach nur noch kleine Gewinne. 15.000–20.000 Iterationen reichen.
- `feet_slide` ist im Reward ca. 30-mal schwächer als `action_rate` und wird praktisch ignoriert.
- Stöße (`push_robot`, alle 4–8 s) kamen kaum vor: Der Timer startet bei jedem Reset neu, und die Episoden dauern seit dem Truncation-Fix nur ca. 4 s. Stöße gab es also nur selten und dann spät in der Haltephase.

### 2.6 Deine Beobachtungen

- **„Er rutscht auf dem Boden“:** bestätigt im Ergebnis, aber die Ursache ist eine andere als gedacht. Er gleitet kaum. Er steht auf den Fußkanten, weil er die Fußgelenke gegen den Anschlag drückt, dreht sich auf der Kante, trippelt und wandert dauernd.
- **„Zwischen den Armen statt mit den Händen“:** Technisch berühren nur die Hände den Ball. Inhaltlich hast du recht: Er fängt nicht mit den Händen vor dem Körper, sondern klemmt den Ball mit hohem Druck an die linke Brust.

## 3. Vorschläge, gewichtet

Wirkung: erwarteter Effekt auf Robustheit am echten Roboter und auf natürliches Verhalten (★ bis ★★★★★). Sicherheit: wie gut die Wirkung durch die Messungen belegt ist. Aufwand: Code und Training.

| Rang | Vorschlag | Wirkung | Sicherheit | Aufwand |
|---|---|---|---|---|
| 0 | Bewertungs-Suite ins Repo (Voraussetzung) | indirekt hoch | hoch | klein |
| 1 | Gelenkanschlag- und Drehmoment-Exploit beseitigen | ★★★★★ | hoch | klein |
| 2 | Episoden wie der echte Ablauf: langes, zufälliges Warten, Halten nach dem Fang, Stillstand belohnen | ★★★★★ | hoch | mittel |
| 3 | Latenz messen und breiter trainieren (plus Aktor-Delay) | ★★★★ | mittel | klein (plus Labor-Messung) |
| 4 | Stöße wirksam machen | ★★★ | hoch | sehr klein |
| 5 | Exploration dämpfen (Entropie) | ★★★ | mittel | sehr klein |
| 6 | Bereitschaftshaltung und Kniebeuge begrenzen | ★★★ | mittel | klein |
| 7 | Fangstil und Endeffektor festlegen und modellieren | ★★★★ | mittel (Entscheidung nötig) | mittel bis groß |
| 8 | Unified mit Unterhand-Clips (Phase 1 als Vorprüfung) | ★★★★ | mittel (blockiert) | groß |
| 9 | Sim2Sim in MuJoCo früh testen | ★★★ | mittel | mittel bis groß |
| 10 | Rekurrente Policy (GRU) | ★★ | niedrig bis mittel | mittel |
| 11 | AMP-Stil-Prior aus den Clips | ★★★ | niedrig | groß |
| 12 | Lauf-Policy als Warmstart oder hierarchisch | ★★ (jetzt) | niedrig (jetzt) | groß |

### 0. Bewertungs-Suite ins Repo

- **Was:** `eval/eval_catch.py` und `eval/run_all.sh` als `scripts/rsl_rl/eval_catch.py` übernehmen. Damit gibt es Fangquote, Füße, Haltung, Haltegeometrie und Drehmomente unter allen acht Szenarien mit einem Befehl.
- **Warum:** Die Fangquote im Training sagt fast nichts über die eigentlichen Schwächen. Jede der folgenden Änderungen sollte daran gemessen werden, vor allem am Szenario „langes Warten“, an der Fußneigung und an der Drehmoment-Sättigung. Das ist zugleich die fehlende feste Auswertung aus Punkt 6 der offenen Punkte in der Spec.

### 1. Gelenkanschlag- und Drehmoment-Exploit beseitigen (alle drei Tasks)

- **Was:**
  - Soll-Winkel auf den (weichen) Gelenkbereich klemmen, z. B. im Action-Term nach Skalierung und Offset. Alternativ grob `clip_actions` im Runner setzen.
  - Drehmoment-Strafe `joint_torques_l2` in Phase 2 ergänzen, Startwert wie Unified (−1e-5), und/oder eine Strafe für Drehmoment über ca. 80 % des Limits.
  - Fußflach-Term: Neigung der Füße bei Bodenkontakt bestrafen, über die projizierte Schwerkraft der `ankle_roll_link`s.
  - Optional `dof_pos_limits` deutlich stärker.
- **Warum:** Das ist der am klarsten belegte Fehler (Abschnitt 2.2 und 2.4) und erklärt mehrere Symptome auf einmal: Stehen auf der Fußkante, Drehen und Trippeln, überkreuzte Beine und das Einklemmen mit Maximalkraft. Am echten G1 würde dauerhaft gesättigtes Fußgelenk-Drehmoment die Motoren schnell überhitzen und die Endanschläge belasten. Auf der Kante stehen ist auf echtem Boden instabil. Nebeneffekt: `last_action` wird unnormalisiert beobachtet (`empirical_normalization=False`), mit Werten bis ca. 9. Begrenzte Aktionen machen auch die Eingaben sauberer.
- **Risiko:** Die Fangquote kann zunächst sinken, weil die Policy einen neuen Stand und einen sanfteren Griff finden muss. Ein neuer Lauf von null ist sinnvoller als Weitertrainieren, da der alte Checkpoint auf den Exploit eingespielt ist.

### 2. Episoden wie der echte Ablauf

- **Was:**
  - Wartezeit vor dem Wurf zufällig und lang, z. B. 0,5–8 s, mit Schwerpunkt auf kurzen Zeiten, damit das Lernen nicht zu langsam wird.
  - Nach dem bestätigten Fang die Episode nicht beenden, sondern 2–4 s weiterlaufen lassen: weiter Halte-Reward, Stillstand belohnen, Fang-Bonus nur einmal pro Episode. Danach Ende als Truncation.
  - Stillstand-Terme vor dem Wurf und beim Halten: Becken-Geschwindigkeit und Drehrate, Fußgeschwindigkeit bei Kontakt, Abheben der Füße bestrafen.
  - Später mehrere Würfe pro Episode.
- **Warum:** Am echten Roboter wartet er immer unbestimmt lange. Im Test führt das zu 28 % Fängen und 28 % Stürzen. Auch das Halten nach dem Fang hat er nie gelernt und läuft dabei 4 m weit. Das ist für den Einsatz am Roboter der größte Abstand zwischen Training und Realität.
- **Hinweis:** Der Truncation-Fix bleibt wichtig. Wenn die Episode nach dem Fang weiterläuft, gibt es keinen Anreiz mehr, den Fang hinauszuzögern, solange der Bonus nur einmal fällt. Für Unified gilt dasselbe: Der Wurf ist dort an den Clip gekoppelt, die Wartezeit also ebenfalls kurz und fest.

### 3. Latenz messen und breiter trainieren

- **Was:**
  - Die echte Latenz VICON → MQTT → Roboter messen (Spec, offener Punkt 4).
  - `obs_delay_steps_range` im Default auf mindestens den gemessenen Wert plus Reserve setzen, z. B. (1, 6). Wichtig: Der Ringpuffer wird aus dem Default dimensioniert, eine Curriculum-Änderung allein genügt nicht.
  - Zusätzlich eine kleine Aktor-Verzögerung (0–1 Steuerschritte) randomisieren.
- **Warum:** Schon 80–120 ms statt 20–60 ms kosten 6 Prozentpunkte. Die echte Kette ist unbekannt und vermutlich eher langsamer als im Training angenommen.

### 4. Stöße wirksam machen

- **Was:** `interval_range_s` auf ca. (1, 3) s setzen oder `is_global_time=True`. Stöße gezielt auch während Flug und Halten.
- **Warum:** Im Training kamen sie kaum vor (Abschnitt 2.5). Im Test kosten Stöße 6 Prozentpunkte durch fallen gelassene Bälle. Gestürzt ist er dabei fast nie, die Grundstabilität ist also gut. Mit Vorschlag 2 werden die Episoden länger, dann greifen Stöße ohnehin häufiger.

### 5. Exploration dämpfen

- **Was:** `entropy_coef` von 0,005 auf ca. 0,001–0,002 senken oder über das Training abklingen lassen. Alternativ die Streuung nach oben begrenzen.
- **Warum:** Die Streuung ist im Lauf gestiegen statt gefallen. Mit ca. 0,9 (in Aktions-Einheiten) trainiert die Policy mit sehr verrauschten Bewegungen. Das begünstigt grobe, rauschfeste Strategien wie die tiefe Hocke und das Einklemmen gegenüber präzisen. Kleiner, billiger Eingriff, der Effekt ist aber weniger sicher als bei 1 und 2.

### 6. Bereitschaftshaltung und Kniebeuge begrenzen

- **Was:** Vor dem Wurf eine Bereitschaftspose belohnen: Gelenke nahe der Grundhaltung, Hände vor dem Bauch, Arme nicht abgespreizt. `base_height` nicht unter ca. −5 abbauen oder die Kniebeuge direkt begrenzen.
- **Warum:** Er wartet in 72° Kniebeuge statt 38° und mit abgespreizten Armen. Am echten Roboter heißt das hohe Dauerlast in den Kniemotoren während jeder Wartezeit. Ein Teil davon verschwindet vermutlich schon durch die Drehmoment-Strafe aus Vorschlag 1.

### 7. Fangstil und Endeffektor festlegen

- **Was:**
  - Mit der Gruppe entscheiden, wie gefangen werden soll und welche Hand am echten G1 sitzt (Gummihand, Inspire-Hand mit ca. 0,88 kg pro Hand, 3D-Druck; Spec, offener Punkt 7).
  - Für einen Fang vor dem Körper: Fangbox nach vorn verschieben (z. B. ab 0,25 m), Torso-Kontakt als „nicht gesichert“ werten oder bestrafen, symmetrische Handposition belohnen.
  - Die echte Hand modellieren: Masse, Kollisionsform, bei Inspire-Händen Handflächen-Kontakt wie im mjlab-Ansatz.
- **Warum:** Der jetzige Stil (Einklemmen an der linken Brust mit 236 N) ist nur in der Simulation bequem. Die mjlab-Gruppe hatte dasselbe Problem: Erst gefangen mit der Handseite, behoben durch Handflächen-Sensoren und eine Fangbox vor dem Torso. Mit 0,88 kg Hand statt 0,17 kg ändert sich die Armdynamik deutlich. Ein Modell mit der falschen Hand überträgt sich schlecht auf den echten Roboter.

### 8. Unified mit den Unterhand-Clips

- **Was:**
  - Clip von Maik zuerst in Phase 1 prüfen: Kann der Roboter ihn physisch nachmachen? Beine prüfen.
  - Dann `CATCH_MOTION_TIME` setzen und den Wurf auf den Fangpunkt der Referenz zielen lassen (Review-Befund M2).
  - Unified trainieren und mit `--finetune` in Phase 2 verfeinern.
  - In Stufe B einen Teil der Stil-Rewards behalten, sonst driftet die Policy vom menschlichen Stil weg. Das ist die Erfahrung der mjlab-Gruppe.
- **Warum:** Die Referenz liefert Fußstellung, Armhaltung und Fangposition eines Menschen. Damit wäre vieles aus 1, 6 und 7 schon im Ansatz gelöst. Wirkung hoch, aber blockiert durch die Clips und mit eigenen Baustellen. Vorschlag 1 ist trotzdem nötig, weil Unified dieselben unbegrenzten Aktionen hat.
- **Blockade umgehen:** Laut `H-ReACT/MotionBricks_Relevanz.md` (Idee 1 und 2) gibt es im offenen Datensatz BONES-SEED fertige G1-Clips, die einem beidarmigen Fang vor dem Körper nahekommen (`toss_baby_up_R`, `catch_playing_R`). Mit Kimodo ließen sich außerdem Clips mit vorgegebenem Fangpunkt und Fangzeitpunkt erzeugen. Damit kann Unified starten, ohne auf Maiks Aufnahmen zu warten, und M2 und `CATCH_MOTION_TIME` lösen sich, weil Fangpunkt und Fangzeitpunkt dann feststehen. Die Lizenz vorher mit den Betreuern klären.

### 9. Sim2Sim in MuJoCo früh testen

- **Was:** Nach Vorschlag 1 die Policy in MuJoCo mit gleichem Ball und gleicher Wahrnehmung laufen lassen (Spec, offener Punkt 9).
- **Warum:** Stehen auf Endanschlägen und Einklemmen sind genau die Art Verhalten, die an der Kontaktphysik einer bestimmten Engine hängt. Ein früher Engine-Wechsel zeigt solche Abhängigkeiten, bevor es auf den echten Roboter geht.

### 10. Rekurrente Policy (GRU)

- **Was:** Ein GRU-Actor statt 5 Frames Historie (0,1 s).
- **Warum:** Hilft bei verdecktem Ball in der Fangbox, bei Latenz und bei langem Warten, weil sich die Policy an den Wurf erinnern kann. Auch die mjlab-Gruppe schlägt das vor. Erst sinnvoll, wenn Vorschläge 2 und 3 nicht reichen.

### 11. AMP-Stil-Prior aus den Clips

- **Was:** Ein Diskriminator, der menschlich aussehende Bewegung belohnt, ohne dem Clip zeitlich exakt folgen zu müssen.
- **Warum:** Passt gut zu zufällig getimten Würfen und langem Warten, also besser als starres Tracking. Großer Umbau, als Forschungsoption, falls Unified am Timing scheitert. rsl_rl hat AMP nicht im Kern, der Diskriminator müsste ergänzt werden. Die Gruppe hat AMP-Erfahrung aus LocoMuJoCo. Als Daten kommen die BONES-SEED-Clips in Frage (siehe `MotionBricks_Relevanz.md`, Idee 4).

### 12. Lauf-Policy als Warmstart oder hierarchisch

- **Gegen „jetzt“:**
  - Gewichte direkt übernehmen geht nicht: Die Lauf-Policy hat andere Eingänge (der Fang-Actor hat 515) und einen anderen Action-Scale, was die Invariante „gleicher Action-Scale überall“ bricht.
  - Laufen ist für die aktuellen Würfe nicht nötig: Auch bei ±0,3 m seitlicher Abweichung fängt er 99,5 %.
  - Die Standprobleme sind ein Exploit (Vorschlag 1), keine fehlende Fähigkeit.
- **Wann doch:**
  - Wenn die echten Würfe so stark streuen, dass er Schritte machen muss. Das lässt sich aus den VICON-Ball-CSVs der Aufnahmen messen.
  - Wenn nach Vorschlag 1 und 2 der Stand immer noch schlecht ist.
  - Dann ist eine **hierarchische** Lösung sinnvoller als ein Warmstart: Eine fertige, am G1 erprobte Lauf-Policy steuert die Beine über Geschwindigkeits- und Höhenbefehle, die Fang-Policy Oberkörper und Arme.
  - Günstiger Zwischenschritt: Steh- und Gang-Terme aus dem Lauf-Task übernehmen. Das ist Teil von Vorschlag 1 und 2.
  - Konkrete Variante der hierarchischen Lösung ist der universelle G1-Tracker SONIC als untere Ebene (`MotionBricks_Relevanz.md`, Idee 6). Das ist ein hohes Risiko und ersetzt die eigene Policy weitgehend, eignet sich also eher als Vergleichs-Baseline als als Hauptweg.

## 4. Empfohlene Reihenfolge

1. **Paket A, nächster Lauf (ca. 15.000–20.000 Iterationen):** Vorschläge 0, 1, 2, 4, 5, 6 und der Konfigurations-Teil von 3. Wenn beide Workstations frei sind, zwei Läufe parallel, damit sich die Wirkung zuordnen lässt:
   - A1 nur mit Vorschlag 1
   - A2 mit dem ganzen Paket

   Beide mit der Suite auswerten, Schwerpunkt auf „langes Warten“, Fußneigung, Drehmoment-Sättigung und Fangquote.
2. **Paket B, parallel ohne GPU:** Fangstil und Hand mit der Gruppe festlegen (7), Latenz im Labor messen (3), die Streuung echter Würfe aus den VICON-CSVs bestimmen (wichtig für 12).
3. **Paket C, sobald Clips da sind** (Maiks Unterhand-Clips oder vorher schon BONES-SEED): Phase 1 → Unified → Stufe B (8). Danach je nach Ergebnis 9, 10, 11 oder 12.

## 5. Was nach den Messungen nicht nötig ist

- **Reibungs-Randomisierung ändern:** kein Einfluss auf das Rutschen.
- **Laufen für seitliche Abweichungen bis ±0,3 m:** wird schon gefangen.
- **30.000 Iterationen:** ab ca. 15.000–20.000 kaum noch Gewinn.

## 6. Nebenbei

- Die Umgebungsvariable für `UNITREE_MODEL_DIR` (Default = Laptop-Pfad) kann auf `main`. Dann braucht es den Workstation-Branch und das Rebasen nicht mehr (Review-Befund L2).
- Mit der mjlab-Gruppe (Andy, Phillip, Sarah) abstimmen. Sie trainieren denselben Fang-Task mit Inspire-Händen, Handflächen-Kontakt und einem Unitree-Deploy-Adapter und berichten nahezu 100 % nach 6 h auf einer RTX 4080. Austausch lohnt sich in beide Richtungen: eure Wahrnehmungs- und Latenz-Modellierung sowie diese Belastungstests, ihre Hand- und Deploy-Lösung. Und am Ende muss entschieden werden, welche Pipeline auf den Roboter geht.

## 7. Messmethode und Vorbehalte

- Alle Werte stammen aus der deterministischen Policy in der Play-Config. Im Training (stochastisch) ist das Verhalten unruhiger.
- **Fußneigung:** Winkel zwischen der z-Achse des `ankle_roll_link` und der Senkrechten bei Bodenkontakt > 50 N. In der Grundhaltung ist der Fuß damit flach, die Gelenkwinkel Hüfte, Knie und Fuß summieren sich zu ca. 0.
- **Gleiten vs. Drehen:** Gleiten heißt Fußgeschwindigkeit > 5 cm/s bei Drehrate < 0,5 rad/s. Der Fußpunkt ist der Ursprung des Fußgelenks, nicht die Sohle. Die Verschiebung pro Standphase enthält also auch Drehen und Abrollen.
- **Drehmomente:** von Isaac Lab auf das Limit geklemmte PD-Momente. Am echten Roboter kommen Temperatur-Effekte und Motorschutz dazu. Die Aussage zu Überhitzung ist eine Erwartung, keine Messung.
- **Torso-Kontaktkraft:** Kontaktsensor am `torso_link`. Bei aufrechtem Roboter kommt sie nur vom Ball, weil der Torso den Boden nicht berührt.
