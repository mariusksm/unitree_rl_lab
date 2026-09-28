# Protokoll

**Fehlt:**
* Marius (Krankschreibung eingereicht), Steffen (Bitte Krankschreibung abgeben!)

---

## Gruppen

### MoCap
*Yasin, Phillip, Steffen*
* Aufnahme über Shogun Post generiert und in den benötigten Dateiformaten auf GitLab hochgeladen.
* Setup so eingerichtet, dass beim Start des Skripts Shogun Live, Manus und die Ballparabel aufgenommen werden und diese dann mittels Post zusammengesetzt werden. Idee: Vielleicht ein neuer Ball, mal schauen, ob er Hände die aneinanderhaut.

### Event-Camera
* Erstmal beiseitegelegt, bis der Objektivfilter kommt.

*Phillip, Simon, Yakup*
* Haben ein Skript gebastelt, um die Vicon-Daten zu einer Parabel zu verarbeiten; kontinuierliche Ausgabe der approximierten Parabel und Koordinaten des Balls.

*Niklas*
* Kalman-Filter-Implementierung/-Anpassung für die Ballparabel-Approximation.

### Isaac
*Maik*
* Arme waren am Rumschwanken, das wurde gefixt.
* Ein Modell zu laden hat anfangs funktioniert. Dann wurde mit dem Ball gearbeitet und ein bisschen was an den Rewards angepasst, doch dann ging nichts mehr.
* Lieber eine Aufnahme mit der Parabel und richtigen Würfen usw. Dann sollte das besser funktionieren.

### MuJoCo
*Malte*
* Environments fertigbekommen und in GitLab hochgeladen (im Sinne von Torso- und Ebenenschnittpunkt).

### Retargeting
*Minh, Andy, Sarah, Malte*

* **GMR:**
    * **Andy:** Dateiumformatierung hinbekommen (BVH zu PKL und das dann zu NPZ), doch das Retargeting war fehlerhaft, als unsere Daten statt der Testdaten benutzt wurden.
    * **Minh:** Probleme mit den Achsen beim Retargeting, doch overall geht es anfangs ganz gut; Achsen sind richtig gemappt, doch die Rotation nicht.
    * **Malte:** BVH-Datei genommen und versucht zu retargeten, doch dann gesehen, dass irgendwie etwas nicht ganz richtig mit der Datei ist, und versucht, diese anzupassen.
* **SOMA:**
    * **Sarah:** CSV -> BVH konvertiert, doch das ging nicht gut. Dann direkte BVH ausprobiert und da gab es Probleme, weil das Programm eine andere Ausrichtung hat als unsere Daten. Doch per Blender kann man eine Rotationsmatrix darauf anwenden und diese ändern. Kann das MoCap-Team jedoch auch direkt richtig ausgeben.

---

## Ziele

### Gruppen

#### MoCap
*Yasin, Phillip, Steffen, Yakup*
* Sehr viele Aufnahmen im richtigen Format machen.
* Problem-Fixing.

#### Isaac
*Maik*
* Mit der Parabel zum Punkt laufen.

*Simon*
* Will dort mitwirken / sich einarbeiten.

#### MuJoCo
*Malte*
* Problem-Fixing.

#### Retargeting
*Minh, Andy, Sarah, Malte*
* Problem-Fixing.
