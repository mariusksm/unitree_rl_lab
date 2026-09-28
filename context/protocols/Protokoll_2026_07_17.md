# Protokollvorlage

Protokoll 17.07

Abwesenheit: Niklas, Marius (Urlaub)



## Was wurde gemacht

Yakup:
- Remote access für die Workstation eingerichtet
- Externer Zugriff über Cloudflare
- Anleitung auf Discord hochgeladen

Philipp:
- Hat eine Klausur geschrieben
- Bälle umgeklebt fürs Tagging

Simon:
- Reward Funktionen verändert für das Training
- Roboterüberlebenszeit erhöht sich nach und nach
- Filter bearbeitet wodurch die Tiefenerkennung besser ist (+-5cm Toleranz)

Ivanna, Yasin & Steffen:
- Gedanken gemacht welcher Ball verwendet werden soll fürs Training
- Fußball wurde beklebt und wurde besser erkannt
- Aufnahmen sollen verbessert werden auch unter unterschiedlichen Winkeln

Maik:
- Imitation Learning mit den neusten FBX Dateien
- Roboter macht keine klaren Ballfangbewegungen

Sarah:
- Balldateien von MoCap verwendet fürs Training
- Roboterbewegung und Ballmotion sind in manchen Aufnahmen abweichend
- Unterschiedliche Environments für unterschiedliche Hände aufgesetzt
- Training mit Balldaten läuft stabiler

Andy:
- Roboter trainiert damit diese stehen bleiben
- Es werden keine Bewegungen trainiert

Malte:
- Skript geschrieben um PKL Dateien zu filtern
    - Self-Kollisionen werden rausgefiltert
- Versucht Isaac Lab an der Workstation zu installieren

Minh:
- Retargetting Skript verbessert
    - Kalman Filter
    - Tiefpassfilter
- Ballposition in MuJoCo bestimmt

Marius (Discord Nachricht):
- MoCap Daten in eigene Repo integriert
- Isaac Lab eingerichtet (NVIDIA Bug behoben)

## Was ist zutun

Yakup:
- Weitere Workstation einrichten
- Kamerasetup

Philipp:
- Ballaufnahmen stabilisieren

Simon:
- Mehr RL Training und stehen trainieren

Ivanna, Yasin & Steffen:
- Ball bekleben und rumprobieren
- Property angeben um die Ballposition in den Aufnahmen zu stabilisieren
- Mehr Aufnahmen

Maik:
- Imitation Learning Roboterbewegung verbessern

Sarah:
- Ball im Imitation-Learning geregelt bekommen
- RL Environment schreiben

Andy:
- Trainingsumgebung anpassen
- Workstation einrichten

Malte:
- Isaac Lab auf der Workstation installieren

Minh:
- Werte vom Kalman Filter optimieren
- Workstation einrichten

Marius (Discord Nachricht):
- Imitation Learning Pipeline optimieren, da Training nicht Erwartungen entspricht
- Nach 3rd-party pre trained models recherchieren
