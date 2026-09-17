# Kontrolleret format for `visual_persona`

`visual_persona` er en tekstlig, ikke-identificerende portrætramme. Den er ikke
billedgenerering og må ikke beskrive eller antyde egenskaber ved personen. For at
gøre denne grænse håndhævelig bruger generatoren en lukket dansk grammatik i
stedet for en liste over forbudte ord.

## Sætningsform

Feltet indeholder 2-4 sætninger i denne rækkefølge:

1. `Personen vælger [farvet tøj] og [farvet tilbehør].`
2. `Baggrunden er [baggrund].`
3. Valgfrit: `Lyset er [lys].`
4. Valgfrit: `Rammen er neutral.`

De to sidste sætninger må kun bruges i den viste rækkefølge. Alle punktummer,
kommaer og faste ord skal følge formen. Der må ikke tilføjes andre beskrivelser,
adjektiver eller forklaringer. Farver bøjes efter det valgte navneords køn, for
eksempel `en grøn skjorte` og `et grønt tørklæde`.

## Tilladte valg

### Farver

`blå`, `brun`, `grå`, `grøn`, `hvid`, `lilla`, `orange`, `pink`, `rød`, `sort`,
`turkis` og `gul`, med de nødvendige danske bøjningsformer.

### Tøj

- Fælleskøn (`en`): `bluse`, `cardigan`, `frakke`, `jakke`, `kjole`, `nederdel`,
  `skjorte`, `sweater`, `trøje`, `vest`
- Intetkøn (`et`): `halstørklæde`, `tørklæde`

### Tilbehør

- Fælleskøn (`en`): `broche`, `halskæde`, `hat`, `kasket`, `paraply`, `taske`
- Intetkøn (`et`): `armbånd`, `bælte`, `sjal`, `slips`, `tørklæde`, `ur`

### Baggrund og lys

Baggrunde er `en afdæmpet flade`, `en enkel flade`, `en ensfarvet flade`,
`en lys flade`, `en neutral flade`, `en rolig flade`, `et afdæmpet studie`,
`et enkelt studie`, `et lyst atelier` eller `et roligt atelier`.

Tilladte lysord er `blødt`, `klart`, `dæmpet`, `diffust`, `jævnt` og `roligt`.

## Eksempler

Godkendt:

> Personen vælger en blå skjorte og et grønt armbånd. Baggrunden er en neutral
> flade.

> Personen vælger en rød kjole og en sort taske. Baggrunden er
> et lyst atelier. Lyset er diffust. Rammen er neutral.

Afvist er enhver tekst, der ikke passer til formen, også selv om den ellers kun
indeholder tilsyneladende uskyldige ord. Det omfatter blandt andet nationalitet,
sprog, religion, alder, køn, sex, region, uddannelse, status, hår, øjne, hud,
ansigt, krop, andre fysiske eller følsomme træk, steder og institutioner.
