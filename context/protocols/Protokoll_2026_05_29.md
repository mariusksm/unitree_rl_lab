H-ReACT Protokoll 29.05.2026

Krank/Abwesend: Phillip, Jamal

Moderator: Oliver

Protokollant: Sarah

# Arbeit der Letzen Woche

## Gruppe MoCap: Yasin, Steffen, Ivanna

*   Anzug an der rechten Hand erkennen (so wie Ball)
*   Aufnahmen gemacht: Hand noch nicht gut kalibriert
*   Ca. 10 Würfe aufgenommen
*   Hand flackert noch à Handschuhe waren nicht lange verfügbar

## Gruppe Organisation/Pipeline: Marius

*   Unitree Repo angeschaut und läuft
*   TODO: Integration unserer Daten
*   TODO: LATENT paper auf Repo anwenden

## Gruppe Isaac: Maik

*   Sim-to-Sim zum Laufen bekommen
*   Umgebung gebaut, wo Roboter zum Ball gehen soll
*   Training von Modellen à nicht ganz erfolgreich
*   TODO: Reward anpassen?

## Gruppe Loco MuJoCo IL: Sarah

*   Kurzes IL-Training mit Loco MuJoCo zum Laufen mit allen 3 verfügbaren Algorithmen (Mimic, AMP, GAIL) à Training funktioniert ohne Fehler, aber auf kurze Zeit noch keine guten Ergebnisse
*   Umwandlung von Loco MuJoCo Modell (.pkl) zu einem Stable Baselines 3 Modell (.zip), das dann mit RL weiter trainiert werden kann (mit Gymnasium) Environment
*   Dokumentation der aktuell verwendeten Action und Observation Spaces in der README
*   Hinzufügen Ball-Fang LocoEnv
*   Anschauen der .npz Dateien, die von Loco MuJoCo verwendet werden

## Gruppe MuJoCo Ball Trajectory: Malte

*   Trajectories in Gymanisum Environment einbauen à erweitert Obs Space um 8? Werte: eine weitere Ball Position
*   Umwandlung Quaternion?
*   TODO: get observation anpassen

## Gruppe Loco MuJoCo MoCap Daten einbinden: Andy

*   Umwandlung zu .npz
*   Untersuchung von .npz Dateien
*   Loco Mujoco custom trajectory Generation angeschaut und in python zum Laufen gebracht

## Gruppe Event-Kamera: Yakup, Phillip, Simon

*   Flickern weg bekommen
*   Schnittstelle Prophecy angeschaut
*   Gerade funktioniert Live-Ansicht noch nicht
*   Ball Tracking mit Hintergrund Rauschen à Filtert zum Teil zu viel raus
*   Genereller durch zufälligen Startpunkt
*   Jonglieren zeigt noch einige Probleme

## Gruppe Ball-Parabel Vorhersage (Kalman): Niklas

*   Erweiterung Kalman-Filter Ballwurf auf 3D

## Gruppe Retargeting: Minh

*   General Motion Retargeting (GMR) Installation und ausprobiert mit MoCap Daten
*   Roboter bewegt sich seltsam
*   Punkte in Roboterkoordinaten?
*   Arme links-recht vertauscht?
*   Angefangen zu Debuggen

# Aufgaben bis zur nächsten Woche

Meeting nächste Woche über Discord

## Gruppe MoCap:

*   Qualität & Kalibrierung anschauen è Research
*   Eher wenig bis nächste Woche möglich, da Handschuhe nicht verfügbar

## Gruppe Event-Kamera + Kalman:

*   Ground Truth Daten aufnehmen & vergleichen
*   Live-Ansicht zum Laufen bekommen
*   Kalman auch abgleichen mit MoCap

## Gruppe Isaac: Maik

*   Actions anpassen, Training von Modellen
*   Evtl. mit statischen Ball ausprobieren hinzulaufen

## Gruppe Isaac: Marius

*   LATENT paper übertragen auf Unitree Repo

## Gruppe MuJoCo Ball Trajectory: Malte

*   Environment fertigbekommen

## Gruppe Retargeting: Minh

*   GMR Fehler beheben
*   Csv Dateien als Eingabe & **Ausgabe**, wenn Zeit ist

## Gruppe MuJoCo MoCap Retargeting: Andy, Sarah

*   Retargeting recherchieren csv à csv
*   Für MuJoCo csv à npz