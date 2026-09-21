# Danske personaattributter

Du skaber én detaljeret, syntetisk dansk persona i ét samlet JSON-svar. Returnér
kun gyldigt JSON efter det leverede skema, og returnér alle skemafelter. Brug null
for nullable felter, når en værdi ikke findes.

Inputfeltet `job_function` er den officielle jobfunktionsetiket. Feltet
`origin_country_da` indeholder den præcise danske værdi, som skal indgå naturligt i
personaen, for eksempel `Han kommer fra Rumænien`. Omtal aldrig værdien som et felt,
metadata, en etiket eller en kategori. Brug den ikke til at udlede kultur,
nationalitet, etnicitet, udseende, beskæftigelse, navn eller interesser. Feltet
`allowed_job_titles` er den komplette, lukkede liste over tilladte titler for
jobfunktionsetiketten. Hvis listen er tom, skal `job_title` være null. Ellers skal
`job_title` være præcis én værdi fra listen, inklusive stavning og mellemrum.

Regler for de strukturerede felter:

- Bevar de faste demografiske og OCEAN-input uden at ændre dem.
- Skriv 3-6 korte, forskellige færdigheder og 3-6 konkrete, forskellige interesser.
- Hver interesse skal være en dansk fællesnavnefrase med små bogstaver og uden
  afsluttende tegnsætning.
- Interesser må ikke være OCEAN-personlighedstræk eller formuleringer som `kan være
  rolig`.
- Skriv et konkret, syntetisk karrieremål eller en anden fremtidsdrøm, når det er
  naturligt. Brug ellers null.
- Brug `current_status` som den nøjagtige aktuelle status, når den er leveret.
- Færdighederne er syntetiske muligheder, ikke dokumenteret erfaring.
- Variér emnerne på tværs af personer; brug ikke automatisk standardkombinationen
  læsning, gåture og madlavning.

Alle uddannelses-, arbejds-, fritids- og familiedetaljer ud over inputtet er
opdigtede dele af den syntetiske persona. De må gerne være konkrete, men må ikke
ligne dokumentation om en virkelig person. Brug ingen efternavne, rigtige
arbejdsgivernavne, præcise adresser, kontaktoplysninger eller administrative numre.
Udled ikke religion, etnicitet, helbred, seksualitet, politisk overbevisning,
kriminalhistorik eller fysisk udseende. Undgå stereotyper baseret på køn, alder,
kommune, værdien i `origin_country_da`, uddannelse eller arbejdsstatus.
