# Dansk persona- og attributprompt

Du skaber én detaljeret, syntetisk dansk persona i ét samlet JSON-svar. Returnér
kun gyldigt JSON efter det leverede skema, og returnér alle skemafelter. Brug null
for nullable felter, når en værdi ikke findes.

## Input og faste oplysninger

Inputfeltet `job_function` er den officielle jobfunktionsetiket. Feltet
`origin_country_da` indeholder den præcise danske værdi, som skal indgå naturligt i
personaen, for eksempel `Han kommer fra Rumænien`. Omtal aldrig værdien som et felt,
metadata, en etiket eller en kategori. Brug den ikke til at udlede kultur,
nationalitet, etnicitet, udseende, beskæftigelse eller interesser.

Feltet `allowed_job_titles` er den komplette, lukkede liste over tilladte titler for
jobfunktionsetiketten. Hvis listen er tom, skal `job_title` være null. Ellers skal
`job_title` være præcis én værdi fra listen, inklusive stavning og mellemrum.

Bevar alder, `han` eller `hun`, kommune, `origin_country_da`, jobtitel, status og
`marital_status` samt de faste demografiske og OCEAN-input uden at ændre dem. Brug
`current_status` som den nøjagtige aktuelle status, når den er leveret.

## Strukturerede felter

- Skriv 3-6 korte, forskellige færdigheder og 3-6 konkrete, forskellige interesser.
- Hver interesse skal være en dansk fællesnavnefrase med små bogstaver og uden
  afsluttende tegnsætning.
- Interesser må ikke være OCEAN-personlighedstræk eller formuleringer som `kan være
  rolig`.
- Skriv et konkret, syntetisk karrieremål eller en anden fremtidsdrøm, når det er
  naturligt. Brug ellers null.
- `first_name` skal være personaens almindelige fornavn.
- Vælg `current_relationship_status` som `partnered` eller `not_partnered`. Ved
  `partnered` skal `partner_first_name` og `partner_gender` udfyldes; ved
  `not_partnered` skal begge være null.
- `marital_status` er juridisk status og er separat fra det aktuelle forhold:
  `married_or_separated` kræver `legal_status_detail` som `married` eller
  `separated`, og den valgte status skal fremgå af personaen. `married` kræver
  `partnered`, mens `separated` kan have begge aktuelle forholdsstatusser.
  `never_married` betyder ikke, at personaen er single, og skal omtales som aldrig at
  have været gift. `divorced` og `widowed` skal omtales som henholdsvis skilt eller
  enke/enkemand, men personaen kan stadig være `partnered` eller `not_partnered`.
- Partnerkombinationer af samme køn, herunder male/male og female/female, er
  udtrykkeligt tilladt. Brug partnerens navn og ord som mand eller kvinde til at
  afspejle `partner_gender`, men skriv eller udled aldrig en orienteringsbetegnelse.
- Færdighederne er syntetiske muligheder, ikke dokumenteret erfaring.
- Variér emnerne på tværs af personer; brug ikke automatisk standardkombinationen
  læsning, gåture og madlavning.

## Personaen

Skriv én sammenhængende, naturlig og specifik dansk tekst i JSON-feltet `persona`.
Teksten skal være 5-7 sætninger og cirka 350-650 tegn. Den skal læses som et lille
portræt af et menneske, ikke som en opremsning af databasefelter.

Brug værdien fra `origin_country_da` naturligt i sætningen, som fx:

- Ved værdien `Rumænien`: `Han kommer fra Rumænien.`
- Ved værdien `Danmark`: `Hun er fra Danmark.`

Skriv aldrig om inputfelter, metadata, etiketter eller kategorier i personaen.
Indarbejd mindst to af de genererede interesser og mindst én af de genererede
færdigheder naturligt i teksten. Du må bøje, omskrive og sætte dem ind i en større
sammenhæng, så sproget bliver grammatisk korrekt og flydende, når betydningen
bevares. Hvis `career_goals_and_ambitions` ikke er null, skal ambitionen formidles
naturligt, men den behøver ikke gengives ordret. Lad mindst én formulering fra
`allowed_personality_tendencies` inspirere en tydelig, nuanceret beskrivelse af
personligheden. Den må gerne omskrives, så længe betydningen bevares, og OCEAN ikke
fremstilles som en diagnose eller en sikker sandhed. Prioritér altid naturligt dansk
frem for ordret genbrug af de genererede formuleringer.

Gør de syntetiske hverdagsdetaljer konkrete og indbyrdes konsistente:

- Giv personaen et almindeligt fornavn.
- Beskriv en plausibel type arbejdsplads og dens by eller område uden at bruge navnet
  på en virkelig virksomhed.
- Beskriv fritiden gennem konkrete aktiviteter, steder eller fællesskaber.
- Beskriv en syntetisk civil- eller familiesituation. Fornavne på partnere eller
  børn er tilladt, men efternavne er ikke.
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
`origin_country_da` må heller ikke bruges til at udlede nationalitet, job, interesser
eller personlighed.
Brug aldrig `vedkommende`, `personen` eller bare `person`, formuleringer med `kan ...
være` eller `ungdoms- eller erhvervsuddannelse`. Skriv ikke tekniske feltnavne,
datamodeller eller rå OCEAN-scorer. Returnér kun JSON efter skemaet.
