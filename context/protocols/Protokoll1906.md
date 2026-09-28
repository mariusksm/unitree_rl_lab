Protokoll 19.06 

Abwesenheit: Niklas - hat sich den Ellenbogen gebrochen ist gerade im KH

                            Was wurde gemacht

Minh: Retargeting FBX 
    relativ solide Performance auf neusten Beispielen vom 16.06
    Erste Aufnahme super
    Retargeting funktioniert mit flimmernden Kreuzen
    Hand ist nicht in FBX Datei - FBX anscheinend 30 Hz -> Mocap Gruppe

Malte:
    Import funktioniert
    unsere BVH Dateien sind "Nucof" sollen "Lafan"
    Unsere Dateien funktionieren nicht weil "Nucof" im GMR Framework
    Debugging fehlgeschlagen

Andy:
    GMR Problem mit BVH Datei
    Retargeting - Beine versetzt, unnatürliche Armbewegungen, aber funtkioniert eingeschränkt mit Beispieldateien. Viel Gittern besonders rechte Hand.

Sarah: 
    Soma Retargeting BVH to CSV mit neuer Root, welche den Roboter rotiert und auf den Boden versetzt. Mesh verdreht, Retargeting funtkioniert nicht. 
    Verbesserung von Mesh und Körper im Gesamtkontext
    Rotation muss um Ursprung nicht Axen

Mocap (Ivanna, Yasin(Löwe), Steffen):
    Aufnahme 300 Aufnahmen, davon 140 ohne Ball CSV
    circa 160 Aufnahmen ohne komplette Bereinigung
    teilweise flimmert das Handgelenk - Muss noch geklärt werden woran das liegt.
    Anpassung des Workflows zur Automatisierung des Exports der Daten in Shogun Post über Shogun Live und Manus Core. 
    Möglichkeits des Exports in C3D, BVH und FBX, sowie variabler Framerate

Marius:
    Anforderungsübertrag ins Unitry Repo
    Pipeline erstellt für Learning
    Reinforcement Learning: Observations, Rewards 

Maik:
    Trajectory Fitting - Roboter läuft zur Aufprallposition - definiert als 1 Meter über dem Boden.
    Ball wird mit Varainz geworfen: Bei ungenauem Wurf, also erwartete Aufprallposition etwa neben oder hinter dem Roboter, verhält sich der Roboter unmenschlich und kriegt Rotation nicht hin oder ist nicht schnell genug. Teilweise blockiert er den Wurf auch mit seinem Körper.
    Ziel ist die Optimierung des Laufverhaltens.

Simon: 
    sucht neue Gruppe
    PC zu schlecht für Imitation Learning, hat paar Sachen probiert aber ohne Erfolg geblieben

Philipp:
    GMR - Beine sind verschränkt, Finger stecken ineinander 
    Probleme mit Achsenrotation - selbe wie andere GMR Gruppe

Yakup: 
    General Motion Retargeting Setup - Roboter ist stuck in der Erde
 
                              Was ist zutun
Minh:
    Versucht weiter flimmern zu beheben
    Optimiert Retargeting vor allem Achsen

Malte:
    Metriken für Retargeting GMR - Automatischer Check als Metrik für Retargeting Qulaität

Andy:       
    Weitere Bearbeitung der GMR Pipeline mit BVH Dateien
    NPZ Datei von Minh für Mujoco nutzbar machen

Sarah:
    Imitation Learning - Lernen der Hände fixen

Mocap:
    Weitere Aufnahmen
    Fixen des Gittern der Hände

Marius:
    Zweite Trainingsphase (Reinforcement Learning) durchmodellieren
    Vor allem Hände integrieren

Maik:
    Default Roboter Position als vor dem Ball 
    Fixe Position des Werfers!
    Weitere Optimierung der Beinbewegungen

Philipp & Yakup (Mocap):
    Script zur Implementierung des Qualitätsmaß (mittels erster Ableitung zum Beispiel)
    Filterung der von Post exportierten Daten (Zero-Lag Filter ohne Phasenverschiebung)

Simon:
    Reward Functions überlegen für Reinforcement Learning - anfängliches Mujoco testing



TODO: FÜr jeden Pipeline Schritt automatische Prüfung mittels Parametern:
    Soll vs Ist vergleich mit Abstandsmaß
    Zittern = Erste Ableitung wechselt Ganzheitlich
    Abstand Original zu Retargeted, um Qualität zu messen
    

    