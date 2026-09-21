# Dansk persona- og attributprompt

Du skaber én detaljeret, syntetisk dansk persona i ét samlet JSON-svar. Returnér
kun gyldigt JSON efter det leverede skema, og returnér alle skemafelter. Brug null
for nullable felter, når en værdi ikke findes.

## Input og faste oplysninger

Inputfeltet `job_function` er den officielle jobfunktionsetiket. Feltet
`origin_country_da` indeholder den præcise danske værdi, som skal indgå naturligt i
personaen, for eksempel `Han kommer fra Rumænien`. Omtal aldrig værdien som et felt,
metadata, en etiket eller en kategori. Brug den ikke til at udlede kultur,
nationalitet, etnicitet, udseende, beskæftigelse, navn eller interesser.

Feltet `allowed_job_titles` er den komplette, lukkede liste over tilladte titler for
jobfunktionsetiketten. Hvis listen er tom, skal `job_title` være null. Ellers skal
`job_title` være præcis én værdi fra listen, inklusive stavning og mellemrum.

Bevar alder, `han` eller `hun`, kommune, den præcise værdi i `origin_country_da`,
leveret jobtitel eller aktuelle status samt de faste demografiske og OCEAN-input
uden at ændre dem. Brug `current_status` som den nøjagtige aktuelle status, når den
er leveret.

## Strukturerede felter

- Skriv 3-6 korte, forskellige færdigheder og 3-6 konkrete, forskellige interesser.
- Hver interesse skal være en dansk fællesnavnefrase med små bogstaver og uden
  afsluttende tegnsætning.
- Interesser må ikke være OCEAN-personlighedstræk eller formuleringer som `kan være
  rolig`.
- Skriv et konkret, syntetisk karrieremål eller en anden fremtidsdrøm, når det er
  naturligt. Brug ellers null.
- Færdighederne er syntetiske muligheder, ikke dokumenteret erfaring.
- Variér emnerne på tværs af personer; brug ikke automatisk standardkombinationen
  læsning, gåture og madlavning.

## Personaen

Skriv én sammenhængende, naturlig og specifik dansk tekst i JSON-feltet `persona`.
Teksten skal være 5-7 sætninger og cirka 350-650 tegn. Den skal læses som et lille
portræt af et menneske, ikke som en opremsning af databasefelter.

Indarbejd værdien fra `origin_country_da` som et almindeligt faktum i naturligt
dansk. Eksempler:

- Ved værdien `Rumænien`: `Han kommer fra Rumænien.`
- Ved værdien `Danmark`: `Hun er fra Danmark.`

Skriv aldrig om inputfelter, metadata, etiketter eller kategorier i personaen. Brug
mindst to af de genererede interesser og mindst én af de genererede færdigheder
ordret. Hvis `career_goals_and_ambitions` ikke er null, skal den formulering indgå
ordret. Brug mindst én præcis formulering fra `allowed_personality_tendencies`
ordret, så personligheden bliver tydelig uden at gøre OCEAN til en diagnose eller en
sikker sandhed.

Gør de syntetiske hverdagsdetaljer konkrete og indbyrdes konsistente:

- Giv personaen et almindeligt fornavn, som ikke vælges ud fra værdien i
  `origin_country_da`.
- Beskriv en plausibel type arbejdsplads og dens by eller område uden at bruge navnet
  på en virkelig virksomhed.
- Beskriv fritiden gennem konkrete aktiviteter, steder eller fællesskaber.
- Beskriv en syntetisk civil- eller familiesituation. Fornavne på opdigtede partnere
  eller børn er tilladt, men efternavne er ikke.
- Nævn en konkret drøm, plan eller ambition, hvis den findes.

Hvis `education_level` er `ungdomsuddannelse eller erhvervsuddannelse`, skal du
vælge præcis én af de to muligheder til den syntetiske persona. Skriv enten
`ungdomsuddannelse` eller `erhvervsuddannelse`, aldrig begge, og tilføj en plausibel
retning og by. For andre uddannelsesniveauer må du tilsvarende tilføje en plausibel
retning og by, men du må ikke nævne en virkelig institutions navn. Uddannelsen er en
opdigtet personadetalje og må ikke fremstilles som observeret kildedata.

## Sikkerhed og syntetiske detaljer

Alle uddannelses-, arbejds-, fritids- og familiedetaljer ud over inputtet er
opdigtede dele af den syntetiske persona. De må gerne være konkrete, men må ikke
ligne dokumentation om en virkelig person. Brug ingen efternavne, rigtige
arbejdsgiver- eller institutionsnavne, præcise adresser, kontaktoplysninger,
administrative numre, CPR-numre, telefonnumre eller links.

Udled ikke religion, etnicitet, helbred, seksualitet, politisk overbevisning,
kriminalhistorik eller fysisk udseende. Undgå stereotyper baseret på køn, alder,
kommune, værdien i `origin_country_da`, uddannelse eller arbejdsstatus. Værdien i
`origin_country_da` må heller ikke bruges til at udlede nationalitet, navn, job,
interesser eller personlighed.

Brug aldrig `vedkommende`, `personen` eller bare `person`, formuleringer med `kan ...
være` eller `ungdoms- eller erhvervsuddannelse`. Skriv ikke tekniske feltnavne,
datamodeller eller rå OCEAN-scorer. Returnér kun JSON efter skemaet.
