Protokoll 05.06.2026

Zu spät: Ivanna (15:07) hatte aber schon Kuchen mitgebracht

Krank/Abwesend: Marius (Angekündigt im Urlaub)

Moderator: Dominik

Protokollant: Simon

# Arbeit der Letzen Woche

## Gruppe Organisation/Pipeline: Marius

## Gruppe MoCap: Yasin, Steffen, Ivanna

- Phillip hat Manus Adapter modelliert und 3D-gedruckt
- Literatur Recherche: Genauigkeit der Aufzeichnung verbessern
- Wieder die Manus Gloves nicht da
- Echtzeitsynchronisation von Manus und Vicon angeschaut

## Gruppe Isaac: Maik

- Modell trainiert, dass Roboter rennt (relativ langsam)
- Arme werden mit benutzt für Stabi (weniger Arme benutzen, weil das wohl gefährlich wäre)
- Dieses Modell als Basis für zukünftige Modelle

## Gruppe Loco MuJoCo IL: Sarah

- Probiert zu trainieren (Semi-Erfolgreich)
- Noch nicht so viel Erfahrung mit den Parametern

## Gruppe MuJoCo Ball Trajectory: Malte

- Parabelfit für Flugbahn
- Berechnung des Treffpunkts anhand einer Parabel

## Gruppe Retargeting: Sarah, Andy, Minh

- (Sarah) Soma Retargeter von NVDEA
- Installation sehr einfach
- Bisher noch falsches Datenformat (akzeptiert nur bvh Daten, nicht csv)
- Skript zum Umwandeln von csv zu npz
- (Andy) GenerellMotion Retargeting ausprobiert
- Umwandlung von pkl zu npz hat nicht funktioniert
- (Minh) GMR weiter angeschaut (Code zu Komplex)
- Pycharm runtergeladen
- Selben Fehler wie letzte Woche mit dem Mapping am Debuggen

## Gruppe Event-Kamera: Yakup, Phillip, Simon

- Programm überarbeitet, dass der Ball bei der Kamera gesucht wird
- Event Kamera Aufnahme mit Ground Truth von MoCap gleichzeitig aufgenommen
- Ground Truth gecleaned
- Problem mit Tiefenschätzung und Anti-Flicker
- Halbe Lösung: teilweise Vicon Kameras ausgemacht

## Gruppe Ball-Parabel Vorhersage (Kalman): Niklas

- Kalman Filter auf Ground Truth Daten von Vicon angewandt
- Berechnet die Flugbahn anhand von Vicon Daten

# Aufgaben bis zur nächsten Woche

## Gruppe MoCap

- Manus + Vicon aufnahmen generieren

## Gruppe Event-Kamera + Kalman

- Event Kamera erstmal auf Eis gelegt (Auf IR-Filter warten)
- Flugbahn über Vicon Daten berechnen und in die Daten Pipeline mit einbauen

## Gruppe Isaac

- Arme weniger schwingen lassen
- Ball in die Scene implementieren
- Roboter Informationen über Geschwindigkeit und Ball geben und Roboter zum Aufprallpunkt laufen lassen

## Gruppe MuJoCo Ball Trajectory

- Modelle mit unterschiedlichen Informationen, welche dem Roboter gegeben werden fertig machen

## Gruppe Retargeting

- (GMR) Konvertierung von pkl zu npz
- Andere Formate anschauen und weitere Möglichkeiten überprüfen
- (Soma) Anschauen, ob man auch csv als Input Möglich ist

## Gruppe Loco MuJoCo IL

- Weiter einarbeiten und mehr Erfahrung mit Parametern sammeln

Für die Zukunft die Ziele als Issue in GitLab erstellen für besseres Tracking.
