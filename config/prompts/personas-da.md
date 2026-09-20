# Danske personabeskrivelser

Du skriver seks korte, syntetiske tekster på naturligt dansk ud fra faste input:
`professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`,
`culinary_persona` og `persona`. Returnér kun JSON efter skemaet. Hver tekst er
sammenhængende prosa på 2-4 sætninger. `persona` er højst 600 tegn og er beregnet til
senere billedprompting; den må ikke være en visuel beskrivelse.

`persona` skal indeholde alle følgende fakta i naturlige sætninger: alder i formen
`[tal] år`, det danske kønsord, den nøjagtige kommune og oprindelsesetiket samt den
brede danske uddannelsesbetegnelse, der er leveret i `education_level`. Gentag den
leverede uddannelsesbetegnelse ordret. Den er en sammenlagt kategori: udled eller
opfind aldrig et mere specifikt uddannelsesniveau eller en uddannelsesinstitution.
Returnér altid alle skemafelter; ingen felter må udelades. For ikke-ansatte er de
aktuelle statusfraser `ledig`, `studerende`, `pensionist` eller `uden for
arbejdsmarkedet`. For ansatte uden
jobtitel er statusfraserne `selvstændig` eller `medarbejdende ægtefælle`. Brug altid den
leverede kanoniske aktuelle statusfrase. Indlejr nøjagtigt
2-3 forskellige interesser ordret fra `generated_attributes`. Medtag 1-2 forsigtige
OCEAN-tendenser fra den lukkede ordliste; brug altid hedging som `kan`, `ofte` eller
`muligvis`. Skriv aldrig punktlister, nummerering, parenteser, klammer eller
semikolonlister.

Regler:

- Bevar demografi, personlighed og godkendte attributter uden at opfinde fakta.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder ud
  over den leverede kommune.
- Udled ikke religion, etnicitet, helbred, seksualitet, politik eller kriminalitet.
- Påstå ikke familie, husstand, diagnose eller fysisk udseende.
- Giv aldrig tidligere eller tidligere formuleret arbejde, heller ikke for ansatte:
  undgå `tidligere`, `førhen`, `arbejdede`, `har arbejdet`, `pensioneret fra` og
  `forhenværende`.
- Behandl OCEAN som svage tilbøjeligheder, aldrig som diagnoser, evner eller mangler.
- Omtal færdigheder som mulige interesser eller styrker, ikke dokumenteret erfaring.
- Brug naturlige danske ord og hold hvert specialiseret felt i sit eget domæne.
- De seks tekster skal være forskellige; gentag ikke en hel tekst ordret.
