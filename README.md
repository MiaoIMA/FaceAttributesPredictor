# 🤖 Gesichtserkennungsbasierte Vorhersage von Alter, Geschlecht und Rasse
# Face Recognition-based Prediction of Age, Gender, and Race

<p align="center">
  <img src="./sample.png" />
</p>


---



---


#### Projektbeschreibung

#### Projekt Hintergrund
In diesem Projekt zielen wir darauf ab, ein Modell zu erstellen, das in der Lage ist, das Alter, Geschlecht und die Rasse einer Person aus ihren Bildmerkmalen in einem Bild vorherzusagen. Die Grundlage des Projekts ist der UTKFace-Datensatz, der über 20.000 Gesichtsbilder mit einem Altersbereich von 0 bis 116 Jahren enthält.

#### Tiefgreifende Diskussion über Daten Vorverarbeitung
In diesem Abschnitt haben wir strenge Vorverarbeitung und Bereinigung des Datensatzes durchgeführt. Insbesondere bei der Verteilung von Alter und Rasse haben wir festgestellt, dass es einige Ungleichgewichte gibt. Zum Beispiel gibt es deutlich mehr Bilder im Alter zwischen 22 und 25 Jahren und auch mehr Bilder von Weißen. Um zu verhindern, dass das Modell diese Kategorien bevorzugt, haben wir eine angemessene Datenbalancierung durchgeführt, indem wir zufällig einige Proben aus diesen Kategorien gelöscht haben. Gleichzeitig haben wir uns aufgrund der geringen Anzahl von Proben dazu entschieden, Daten von Personen über 80 Jahren zu ignorieren.
<p align="center">
  <img src="./Age Distribution.png" />
</p>

<p align="center">
  <img src="./Race_names Distribution.png" />
</p>

<p align="center">
  <img src="./Genders Distribution.png" />
</p>

#### Modellbau und Analyse
In Bezug auf den Modellaufbau haben wir uns für eine relativ einfache CNN-Struktur als unser Grundmodell entschieden. Die spezifische Modellarchitektur ist wie folgt:
```python
agemodel = Sequential()
agemodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Conv2D(64, (3,3), activation='relu'))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Conv2D(128, (3,3), activation='relu'))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Flatten())
agemodel.add(Dense(64, activation='relu'))
agemodel.add(Dropout(0.5))
agemodel.add(Dense(1, activation='relu'))
```
Wir haben bewusst keine beliebten Modelle wie ResNet, DenseNet oder VGG verwendet, da sie viele Parameter enthalten und daher eine lange Trainingszeit benötigen, was auf Plattformen mit begrenzten Ressourcen, wie Kaggle, nicht praktikabel ist.
<p align="center">
  <img src="./model_loss_age..png" />
</p>
#### Herausforderungen und Analyse des Modells
Obwohl unser Modell in gewissem Maße die Vorhersageziele erreicht hat, erkennen wir durch Beobachtung der Loss-Veränderung, dass es noch erheblichen Verbesserungsbedarf gibt und es noch weit von der Optimierung entfernt ist. Im Vergleich zu anderen fortschrittlichen Modellen, wie DeepFace, gibt es immer noch eine erhebliche Lücke in Bezug auf Vorhersagegenauigkeit und Stabilität unseres Modells.

#### Projektausblick
In zukünftigen Arbeiten planen wir, fortschrittlichere und reife Modelle wie DeepFace zu verwenden und zu erforschen, wie sie auf der React-Plattform für die Echtzeitvorhersage von Alter, Geschlecht und Rasse für mehrere Personen implementiert werden können, um die Praktikabilität und Benutzererfahrung unseres Modells weiter zu verbessern. 

Auf dieser Grundlage hoffen wir, dass dieses Projekt zu einer offenen Plattform werden kann, die nicht nur auf die Vorhersage von Alter, Geschlecht und Rasse beschränkt ist. In der Zukunft könnte es auf weitere Gesichtsmerkmalsanalysen, wie Emotionserkennung, Gesichtserkennung usw., erweitert werden, um den Benutzern mehr Dienstleistungen und Bequemlichkeiten zu bieten.

##Einrichtung

```shell
git clone https://github.com/Hyuto/yolov8-tfjs.git
cd yolov8x
yarn install #Install dependencies
```

##Skripte

```shell
yarn start # Start dev server
yarn build # Build for productions
```

## Reference

- https://github.com/ultralytics/ultralytics
- https://github.com/Hyuto/yolov8-onnxruntime-web
