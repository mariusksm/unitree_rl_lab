## Protokoll 22.5

### Yakup

- DBSCAN als Clustering Algorithmus implementiert
- Trigger-Box um Aufnahme zu starten sobald Ball hindurch fliegt für die Eventkamera Aufnahme
- Region of Interest Einstellung für die Event Kamera
- Trajectory Prediction Programm geschrieben: https://gitlab.cc-asp.fraunhofer.de/iml/oe130/projektgruppe-h-react/event-cam-trajectory
- [https://dl.acm.org/doi/pdf/10.1145/3728423.3759404](https://dl.acm.org/doi/pdf/10.1145/3728423.3759404) "Volleyball Paper"
- [https://arxiv.org/pdf/2506.07860](https://arxiv.org/pdf/2506.07860) "Ping Pong Paper"

_Bis nächste Woche:_

- Kamera direkt anbinden um das Tracking Live zu machen

### Phillip

- IR Filter rausgesucht
- Kalman Filter angeschaut
- Event Camera Calibration aufgesetzt [https://gitlab.cc-asp.fraunhofer.de/iml/oe130/projektgruppe-h-react/e2calib-in-a-box](https://gitlab.cc-asp.fraunhofer.de/iml/oe130/projektgruppe-h-react/e2calib-in-a-box "https://gitlab.cc-asp.fraunhofer.de/iml/oe130/projektgruppe-h-react/e2calib-in-a-box")
- Recherche ("Volleyball Paper", "Ping Pong Paper")

### Simon

- "KI Modell" zum tracken vom Ball/beliebigen Objekten
    - funktioniert live nicht wirklich
- DBSCAN implementiert
- Weitere Methoden zum besseren tracken des Balls ausprobiert (Ping Pong Paper, Volleyball Paper)

_Bis nächste Woche:_

- Die anderen Ansätze zum Ball tracken weiter ausprobieren

### Andy

- Loco Mujoco Daten Umwandlung
- SOMA verwendet um Motion Capture Daten in .npz umzuwandeln (noch nicht vollständig)
- Mosh++ (?) verwendet um Daten in ein Mujoco kompatibles format umzuwandeln

_Bis nächste Woche:_

- Umwandlung zum laufen bekommen
- (?) Datenbank anschauen um fertige Mocap Aufnahmen als Beispiel zu verwenden

### Malte

- Latent Paper zum Tennis spielen simulieren gelesen

_Bis nächste Woche:_

- Mujoco Environments modifizieren um Yakup's Parabel Prediction einzubinden

### Maik

- Sim to Sim Transfer von Isaac Sim zu Mujoco
- Simulierter Roboter ist komplett steuerbar
- Policy auf den simulierten Roboter geladen
- Probleme mit dem Controller Input

_Bis nächste Woche:_

- Linux aufsetzen
- Sim2Sim zu testen
- Tennis Paper
- Seperat Oberkörper Trainieren anstatt den kompletten Roboter

### Sarah

- Alle alten sachen im Gitlab gepusht
- Loco Mujoco Environment für den G1 Roboter mit Händen vorbereitet
- Training in diesem Environment mit Mujoco MJX, GPU Probleme
- Conversion von Loco Mujoco zu einem "Stable Baselines" Modell

_Bis nächste Woche:_

- Auf stärkerem Computer Imitation Learning ausprobieren
- Dokumentation schreiben und Code refactoren
- Für Datenumwandlung und/oder Trajectory Prediction mithelfen

### Ivana, Steffen & Yassin

- Skript mit Shogun Live geschrieben, dass den beklebten Unlabeled Marker Ball in der Hand (oder anderem beliebigem Marker vom Mocap Anzug) von der werfenden Person erkennt um diesen als Ball zu erkennen und tracken zu können.  
    Kann als CSV exportiert werden.
- Neuen Shogun Anzug kalibriert, sodass die Aufnahme Qualität deutlich besser geworden ist
- Ball Tracking im Kontext Fußball und Tischtennis researched

_Bis nächste Woche:_

- Aufnahmen erstellen und "Aufnahme Pipeline" optimieren
- Eventuell mit zwei Personen im Anzug den Ball hin und her werfen

### Marius

- In die Codebase vom Unitree Reinforcement Lab eingearbeitet
- "Managerbase Struktur" in Isaac Lab verstanden und angefangen zu visualisieren damit das alle verstehen könnten

_Bis nächste Woche:_

- Codebase weiter researchen sodass diese als Basis verwendet werden kann
- Diagramm auf GitLab hochladen

### Jamal

- In Adversarial Motion Priors eingearbeitet und getestet
- AMP benutzt zwei Reward Signale und arbeitet in zwei Phasen  
    Stall und Discriminator Reward (...)
- Auf Beispieldaten in Isaac Lab simuliert und trainiert
- Flowchart zum Visualisieren vorbereitet (später im Discord)
- Fragt nach ETA für stärkere Computer

_Bis nächste Woche:_

- Zwei Paper lesen zum Thema Adversarial Motion Plannung und Integration

### Niklas

- Kalman Filter angeschaut
- Ballwurf Simulation mit randomisierten Beispiel Daten implementiert um Ball Trajectory vorherzusagen

_Bis nächste Woche:_

- Kalman Filter implementieren
- Weitere mit den Ground Truth Daten experimentieren

### Minh

- General Motion Retargeting um Shogun Live auf den Roboter zu retargeten
- Pipeline um Mocap Skelette in ein Mujoco kompatibles Format umzuwandeln (noch nicht fertig)

_Bis nächste Woche:_

- Installation abschließen
- Neue CSV Dateien in Mujoco verwendbar machen