# Protokoll - 03.08.2026 von Minh Nhat

## Heute nicht anwesend:
- Malte (Urlaub diese Woche und nächste Woche)
- Maik (Urlaub diese Woche)

## Nächste Woche nicht anwesend:
- Sarah

## Nächsten beiden Wochen nicht anwesend:
- Simon (War doch am 10.08.26 da)

## Ab 17.08 im Urlaub:
- Ivanna

## Was diese Woche gemacht wurde:

Yakup, Phillipp: 
- beide Workstations aktiviert
- Kameras von Dominik geholt und in der Halle positioniert, Kabeln verlegt
- Aufnahmen zum Ballfangen gemacht -> ca. ersten 10 Aufnahmen beim Ball funktionieren, restlichen nicht

Phillip:
- Retargetting mit Jans Teleoperation Projekt getestet und augeführt
- Ballpositionen etwas optimiert

Yasin, Steffen, Ivanna:
- Aufnahmen zum Ballfangen mitgemacht

Simon:
- Lern-Algorithmus ppo getestet:
   -> Probleme beim Stehen des Roboters
   -> direkt nach 1-2 Sekunden umgefallen
- dann Lern-Algorithmus sac angeschaut und getestet:
   -> bleibt länger stehen für 10 Sek. fällt aber immer noch um

Marius:
- falsche Daten bemerkt, dann die richtige Daten verwendet
- Korrektheit sichergestellt, dass das Training komplett durchläuft

Sarah:
- Umgebung für das Reinforcement Learning für die verschiedenen Hände implementiert
- Diese Umgebungen wurden auch getestet

Andy:
- An den Parametern rumgeschraubt -> Hände machen mehr die Fangbewegung, aber Roboter zappelt herum
- Dann versucht, Reward-Funktion einzubauen: entweder nur Zappeln oder Springen

Niklas:
- Teleoperation-Projekt von Jan angeschaut und das versucht zu verstehen
- versucht, das Projekt lokal zu testen -> jedoch gab es Probleme mit den Submodulen

Minh Nhat:
- GMR-Projekt mit den neuen Daten getestet
  -> Ballpositionen verschoben, auch hier nur ca. ersten 10 Aufnahmen mit Ball
- Projekt etwas mehr automatisiert, sodass es direkt aus den Recordings-7z-Datei alle zugehörigen Ergebnisse ausgibt


## Nächste Woche

Minh Nhat + Niklas:
- Vorlage für das Zwischenbericht erstellen
- Kleinere Fehler beim Retargeting fixen: Bessere Ballposition + Roboter-Fuß soll nicht mehr teils im Boden stehen

Andy: 
- Rewards anschauen und anpassen
- auch Nullwinkel testen, dass der Roboter nicht mehr stehen bleibt

Sarah: 
- Modell genauer angucken im BallImitation
- Rewards auch beim Imitation anschauen

Yasin, Steffen, Ivanna:
- weitere Aufnahmen erstellen

Simon: 
- das Modell anschauen und Rewards anpasen, damit der Roboter wie ein Mensch stehen kann

Marius:
- versuchen, visuell anschaubare Trainingsresultate zu haben bei der Imitaitonphase
- auch das ganze bei der Workstation trainieren zu lassen

Phillip:
- Änderungen beim Teleoperation-Projekt von Jan hochpushen
- neue Aufnahmen hochladen
- retargettete npz-Dateien hochladen
- Imitation-Learning langsam angucken

Yakup:
- Imitation-Learning langsam angucken