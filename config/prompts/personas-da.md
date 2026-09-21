# Dansk skrivebrief til persona

Skriv én sammenhængende, naturlig og specifik dansk tekst i JSON-feltet `persona`.
Teksten skal være 5-7 sætninger og cirka 350-650 tegn. Den skal læses som et lille
portræt af et menneske, ikke som en opremsning af databasefelter.

Bevar alder, `han` eller `hun`, kommune, den præcise værdi i
`origin_country_da` og leveret jobtitel eller aktuelle status. Indarbejd værdien fra
`origin_country_da` som et almindeligt faktum i naturligt dansk. Eksempler:

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

Hvis `education_level` er `ungdomsuddannelse eller erhvervsuddannelse`, skal du vælge
præcis én af de to muligheder til den syntetiske persona. Skriv enten
`ungdomsuddannelse` eller `erhvervsuddannelse`, aldrig begge, og tilføj en plausibel
retning og by. For andre uddannelsesniveauer må du tilsvarende tilføje en plausibel
retning og by, men du må ikke nævne en virkelig institutions navn. Uddannelsen er en
opdigtet personadetalje og må ikke fremstilles som observeret kildedata.

Brug aldrig `vedkommende`, `personen` eller bare `person`, formuleringer med `kan ...
være` eller `ungdoms- eller erhvervsuddannelse`. Værdien i `origin_country_da` må
ikke bruges til at udlede kultur, religion, nationalitet, etnicitet, navn, job,
interesser, personlighed eller udseende.

Hold teksten ikke-identificerende: ingen efternavne, rigtige arbejdsgivernavne,
præcise adresser, kontaktoplysninger, CPR-numre, telefonnumre, links, helbred,
religion, etnicitet, seksualitet, politik, kriminalitet eller fysisk udseende. Skriv
ikke tekniske feltnavne, datamodeller eller rå OCEAN-scorer. Returnér kun JSON efter
skemaet.
